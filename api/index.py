import os
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
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


class ImagePart(BaseModel):
    image_base64: str
    mime_type: str


class CheckRequest(BaseModel):
    images: list[ImagePart] = Field(..., min_length=1)
    official_text: str = ""
    official_images: list[ImagePart] = Field(default_factory=list)
    reference_url: str | None = ""


class ExtractOfficialRequest(BaseModel):
    images: list[ImagePart] = Field(..., min_length=1)


EXTRACT_OFFICIAL_PROMPT = """
画像はスプレッドシート・表・または公式情報のスクリーンショットです。
表示されている文字をできるだけ正確に転記してください。

ルール:
・推測で補わない。判読できない字は「?」にする
・表は行ごとに改行し、列はタブ（\\t）で区切る
・説明文や前置きは書かない

出力は JSON のみ。形式は次の1オブジェクト:
{"text": "転記した全文（改行は \\n）"}
"""


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
        len(payload.get("images", [])),
        result.get("conclusion", ""),
        result.get("yakki_risk", {}).get("status", ""),
        json.dumps(result.get("rule_violations", []), ensure_ascii=False),
        json.dumps(result.get("fix_points", []), ensure_ascii=False)
    ]]

    service.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range="'Logs'!A:G",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": row},
    ).execute()


BASE_DIR = Path(__file__).resolve().parent.parent

@app.get("/")
def root():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/check")
def healthcheck():
    return {"message": "ok"}


@app.post("/api/extract-official")
def extract_official(req: ExtractOfficialRequest):
    try:
        user_content: list[dict[str, Any]] = [
            {"type": "input_text", "text": EXTRACT_OFFICIAL_PROMPT},
        ]
        for img in req.images:
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{img.mime_type};base64,{img.image_base64}",
                    "detail": "high",
                }
            )

        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
        )

        text = response.output_text
        data = json.loads(text)
        out = data.get("text", "")
        if not isinstance(out, str):
            raise ValueError("モデル出力の text が文字列ではありません。")
        return {"text": out}

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="取り込み結果がJSONではありませんでした。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/check")
def check(req: CheckRequest):
    if not (req.official_text.strip() or req.official_images):
        raise HTTPException(
            status_code=400,
            detail="正式情報は、テキスト貼り付けとスクショのどちらか一方以上を指定してください。",
        )

    try:
        has_text = bool(req.official_text.strip())
        has_shots = bool(req.official_images)

        if has_text and has_shots:
            priority_rule = """
【正式情報の扱い】
・テキスト貼り付けとスクショの両方がある場合: テキストを正（優先）とし、スクショは補助情報とする。
・照合・判定の基準はテキストに従う。スクショは、判読補助・表の構造確認・抜け漏れの再確認に使う。
・テキストとスクショの内容が食い違う場合は、原則としてテキスト側を正式情報として採用し、
  食い違い自体を「要確認」または警告として明示する（どちらが誤りか断定できない場合は要確認）。
"""
        elif has_text:
            priority_rule = """
【正式情報の扱い】
・正式情報はテキスト貼り付けのみ。これを正としてチェック対象画像と照合する。
"""
        else:
            priority_rule = """
【正式情報の扱い】
・正式情報はスクリーンショットのみ。続く「正式情報」画像を読み取り、それを正としてチェック対象画像と照合する。
"""

        user_text = f"""
{priority_rule}
正式情報（テキスト貼り付け）:
{req.official_text if has_text else "（未入力。正式情報はスクショ画像のみ。）"}

参照URL:
{req.reference_url or "なし"}

チェック対象の画像は、このメッセージの後半に続く「チェック対象」として付けた画像です。
複数ある場合はすべて読み取り、ページ間の整合性も含めて照合してください。

出力形式:
{json.dumps(RESPONSE_SCHEMA_EXAMPLE, ensure_ascii=False)}
"""

        user_content: list[dict[str, Any]] = [
            {"type": "input_text", "text": user_text},
        ]
        if req.official_images:
            shot_heading = (
                "=== 正式情報（スクリーンショット・補助。テキスト貼り付けを優先して照合） ==="
                if has_text
                else "=== 正式情報（スクリーンショット） ==="
            )
            user_content.append(
                {
                    "type": "input_text",
                    "text": shot_heading,
                }
            )
            for img in req.official_images:
                user_content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{img.mime_type};base64,{img.image_base64}",
                        "detail": "high",
                    }
                )
        user_content.append(
            {"type": "input_text", "text": "=== チェック対象の画像 ==="}
        )
        for img in req.images:
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{img.mime_type};base64,{img.image_base64}",
                    "detail": "high",
                }
            )

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
                    "content": user_content,
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
    
   # redeploy_env_fix_0319