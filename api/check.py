import os
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

from google.oauth2 import service_account
from googleapiclient.discovery import build

app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

JST = timezone(timedelta(hours=9))

SYSTEM_PROMPT = """
あなたは「画像内テキストの校正・照合・正確性確認」に特化したチェック担当です。
AIは校正者として振る舞い、提案や文章改善は行いません。

目的：
・画像内テキストの誤字脱字
・商品名の完全一致照合（スペース含む）
・価格 / 容量 / 発売日の正確性
・限定条件 / 注釈の抜け
・修正漏れ
・薬機法リスクの検知（※判断ではなく検知）

絶対ルール：
・文章を書き換えない
・ですます調へ変更しない
・デザイン提案をしない
・コピー提案をしない
・最終ページ提案をしない

表記ルール：
・ml / g は半角小文字
・金額の末尾は必ず (税込)
・記号 / () % は半角

薬機法：
・判断ではなくリスク検知のみ
・治る、改善する、効く、消える、再生、修復、細胞レベル、即効、絶対、確実 などは指摘
・商品名に「医薬部外品」が含まれる場合、薬機法に触れる可能性のある表現があれば
  「公式情報と完全一致しているかを最優先で確認すること」と注意喚起する
・判断が難しい場合は「要確認」にする

出力は必ずJSONのみ。
"""

RESPONSE_SCHEMA_EXAMPLE = {
    "conclusion": "OK | 修正必要 | 要確認",
    "comparison": {
        "product_name": "",
        "price": "",
        "volume": "",
        "release_date": "",
        "description": ""
    },
    "rule_violations": [],
    "accuracy": {
        "matches": [],
        "warnings": []
    },
    "yakki_risk": {
        "status": "問題なし | グレー | NG",
        "terms": [],
        "notes": []
    },
    "fix_points": [],
    "space_audit": {
        "official_visible": "",
        "image_visible": "",
        "result": "一致 | 不一致 | 要確認"
    }
}

class CheckRequest(BaseModel):
    image_base64: str
    mime_type: str
    official_text: str
    reference_url: str | None = ""

def get_sheets_service():
    raw = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    creds_info = json.loads(raw)
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("sheets", "v4", credentials=creds)

def append_result_to_sheet(payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    service = get_sheets_service()
    sheet_id = os.environ["GOOGLE_SHEET_ID"]

    row = [[
        datetime.now(JST).isoformat(),
        payload.get("reference_url", ""),
        result.get("conclusion", ""),
        result.get("yakki_risk", {}).get("status", ""),
        json.dumps(result.get("rule_violations", []), ensure_ascii=False),
        json.dumps(result.get("fix_points", []), ensure_ascii=False)
    ]]

    service.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range="Logs!A:F",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": row},
    ).execute()

@app.post("/api/check")
def check(req: CheckRequest):
    try:
        user_text = f"""
正式情報:
{req.official_text}

参照URL:
{req.reference_url or "なし"}

出力形式:
{json.dumps(RESPONSE_SCHEMA_EXAMPLE, ensure_ascii=False)}
"""

        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": SYSTEM_PROMPT}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {
                            "type": "input_image",
                            "image_url": f"data:{req.mime_type};base64,{req.image_base64}",
                            "detail": "high"
                        }
                    ]
                }
            ]
        )

        text = response.output_text
        result = json.loads(text)

        append_result_to_sheet(req.model_dump(), result)
        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="モデル出力がJSONではありませんでした。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))