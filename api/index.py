import os
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Literal
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from google.oauth2 import service_account
from googleapiclient.discovery import build

BASE_DIR = Path(__file__).resolve().parent.parent

_explicit_dotenv = os.environ.get("IMAGE_CHECKER_DOTENV", "").strip()
if _explicit_dotenv:
    _p = Path(_explicit_dotenv)
    _dotenv_candidates = [_p if _p.is_absolute() else (BASE_DIR / _p)]
    load_dotenv(_dotenv_candidates[0], override=False)
else:
    # プロジェクト直下 → api/ → 親フォルダ（例: デスクトップ直下の .env）。未設定キーのみ後続で補う
    _dotenv_candidates = [
        BASE_DIR / ".env",
        BASE_DIR / "api" / ".env",
        BASE_DIR.parent / ".env",
    ]
    for _path in _dotenv_candidates:
        load_dotenv(_path, override=False)

_openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
if not _openai_key:
    _tried = " / ".join(
        f"{p.resolve()}（存在: {p.is_file()}）" for p in _dotenv_candidates
    )
    raise RuntimeError(
        "OPENAI_API_KEY が設定されていません。"
        f" 確認した .env: {_tried}。"
        " 対処: (1) 上記いずれかに `OPENAI_API_KEY=sk-...` と書く（キー名の綴りを確認） "
        "(2) シェルで環境変数 OPENAI_API_KEY を設定する "
        "(3) 別の場所なら IMAGE_CHECKER_DOTENV にパスを設定（相対パスはプロジェクトルート基準）"
    )

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = OpenAI(api_key=_openai_key)

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
・内容部分の日本語としての違和感の検知（下記「日本語違和感チェック」）

日本語違和感チェック（広告・販促向け前提）：
・本コンテンツは広告・販促用のため「体言止め」が多い前提とする。体言止めそのものは誤りとしない。
・「〜です」「〜ます」への統一は求めない（統一を理由とした指摘は禁止）。
・次のような場合に限り、明らかに違和感があるときだけ指摘する（軽微・好みの問題は出さない）。
  - 文法的に不自然な日本語
  - 助詞の誤用（が / を / に / で など）
  - 語順の違和感
  - 意味が曖昧すぎる表現（例：〜な感じ、しっかり、ちゃんと など。ただし広告で意図的に使われていると読み取れる場合は指摘しない）
  - 冗長な表現
  - 一般的に使われない言い回し
  - 読みづらい構文
・広告表現として一般的な省略は許容する。自然な体言止めはOKとする。

日本語違和感の出力ルール：
・書き換え案・推奨表現・コピー案は一切出さない（「どこが」「どの語句・構造が」不自然かを文章で具体的に述べるのみ）。
・指摘は rule_violations または accuracy.warnings のいずれか（または両方）に含める。
・日本語違和感に関する各文字列は、本文中に必ず「日本語違和感」という語を含める（例：先頭を「日本語違和感：」にする。複数ページ時は【全体N枚目】の直後に続けてよい）。
・他の種類の指摘と混在させる場合も、日本語違和感であることが文面から分かるように「日本語違和感」を残す。

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

複数のチェック対象画像があるときは、ユーザーが付与した「チェック画像 全体N枚目」に対応させ、
fix_points・rule_violations・accuracy.warnings（およびページに紐づく accuracy.matches の各項目）の
文字列は必ず【全体N枚目】を先頭に付けて、どの画像（ページ）の指摘か分かるようにする。
日本語違和感も同様に【全体N枚目】を先頭に付け、「日本語違和感」を文面に含める。
複数ページにまたがる場合は【全体2枚目・3枚目】のようにまとめてよい。画像内の商品名が分かれば
【全体N枚目｜商品名略】としてもよい。

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
    "rule_violations": [
        "【全体1枚目】例: 各項目の先頭に【全体◯枚目】。日本語違和感は本配列または accuracy.warnings に「日本語違和感：」を含めて記載（書き換え案は書かない）"
    ],
    "accuracy": {
        "matches": [],
        "warnings": []
    },
    "yakki_risk": {
        "status": "問題なし | グレー | NG",
        "terms": [],
        "notes": []
    },
    "fix_points": ["【全体1枚目】例: 先頭に必ず全体通し番号を付ける"],
    "space_audit": {
        "official_visible": "",
        "image_visible": "",
        "result": "一致 | 不一致 | 要確認"
    }
}


class ImagePart(BaseModel):
    image_base64: str
    mime_type: str


CheckerName = Literal["山田", "栫", "八尋", "川西", "園田", "かわ", "まゆみ", "とみた"]


class CheckRequest(BaseModel):
    checker_name: CheckerName
    images: list[ImagePart] = Field(..., min_length=1)
    first_check_image_number: int = Field(
        default=1,
        ge=1,
        description="このリクエスト先頭の画像が、画面上のチェック対象で何枚目か（1始まり・選択順）",
    )
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

出力は JSON のみ（マークダウンのコードブロックや説明文は禁止）。
形式は次の1オブジェクトだけ:
{"text": "転記した全文（改行は \\n）"}
"""


def _strip_json_code_fence(raw: str) -> str:
    s = raw.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if not lines:
        return s
    lines = lines[1:]
    while lines and lines[-1].strip() == "":
        lines.pop()
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_outer_json_object(raw: str) -> str | None:
    s = raw.strip()
    i = s.find("{")
    if i < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[i : j + 1]
    return None


def parse_model_json_object(raw: str) -> dict[str, Any]:
    """モデルが ```json ... ``` や前置き付きで返しても dict に落とす。"""
    s = _strip_json_code_fence(raw.strip())
    try:
        out = json.loads(s)
    except json.JSONDecodeError:
        frag = _extract_outer_json_object(s)
        if frag is None:
            raise
        out = json.loads(frag)
    if not isinstance(out, dict):
        raise ValueError("モデル出力のトップレベルがJSONオブジェクトではありません。")
    return out


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

    checked_at = datetime.now(JST).strftime("%Y/%m/%d %H:%M")
    row = [[
        checked_at,
        payload.get("checker_name", ""),
        payload.get("reference_url", ""),
        len(payload.get("images", [])),
        result.get("conclusion", ""),
        result.get("yakki_risk", {}).get("status", ""),
        json.dumps(result.get("rule_violations", []), ensure_ascii=False),
        json.dumps(result.get("fix_points", []), ensure_ascii=False)
    ]]

    last_error = None

    for wait in [1, 2, 4]:
        try:
            service.spreadsheets().values().append(
                spreadsheetId=sheet_id,
                range="'Logs'!A:H",
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body={"values": row},
            ).execute()
            return
        except Exception as e:
            last_error = e
            time.sleep(wait)

    raise last_error


@app.get("/")
def root():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/check")
def healthcheck():
    return {"message": "ok"}


@app.post("/api/extract-official")
def extract_official(req: ExtractOfficialRequest):
    response = None
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

        text = response.output_text or ""
        data = parse_model_json_object(text)
        out = data.get("text", "")
        if not isinstance(out, str):
            raise ValueError("モデル出力の text が文字列ではありません。")
        return {"text": out}

    except json.JSONDecodeError:
        raw_out = (getattr(response, "output_text", None) or "") if response is not None else ""
        snippet = (raw_out[:200] + "…") if len(raw_out) > 200 else raw_out
        raise HTTPException(
            status_code=500,
            detail=f"取り込み結果をJSONとして解釈できませんでした。先頭: {snippet!r}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/check")
def check(req: CheckRequest):
    if not (req.official_text.strip() or req.official_images):
        raise HTTPException(
            status_code=400,
            detail="正式情報は、テキスト貼り付けとスクショのどちらか一方以上を指定してください。",
        )

    response = None
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

        n_check = len(req.images)
        start = req.first_check_image_number
        check_index_lines = "\n".join(
            f"・チェック画像 全体{start + i}枚目 … このAPIリクエスト内では上から{i + 1}番目の画像"
            for i in range(n_check)
        )
        numbering_rule = f"""
【チェック画像の通し番号】
画面上でユーザーが選んだ順（上から1枚目・2枚目…）と一致させています。
{check_index_lines}

【指摘の必須ルール】
・fix_points・rule_violations・accuracy.warnings の各要素は、必ず【全体◯枚目】を先頭に付ける（上記の番号を使う）。
・accuracy.matches も、どのページの一致か分かるよう必要なら【全体◯枚目】を付ける。
・comparison の各フィールドは複数ページの要約でよいが、複数商品がある場合は文中に【全体◯枚目】を入れる。
・日本語違和感は rule_violations または accuracy.warnings に入れ、各文に「日本語違和感」を含める。書き換え案・推奨表現は書かない。
"""

        user_text = f"""
{priority_rule}
正式情報（テキスト貼り付け）:
{req.official_text if has_text else "（未入力。正式情報はスクショ画像のみ。）"}

参照URL:
{req.reference_url or "なし"}

{numbering_rule}

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

        text = response.output_text or ""
        result = parse_model_json_object(text)

        sheet_error: str | None = None
        try:
            append_result_to_sheet(req.model_dump(), result)
        except Exception as e:
            sheet_error = str(e)
        if sheet_error:
            result["sheet_log_error"] = sheet_error

        return result

    except json.JSONDecodeError:
        raw_out = (getattr(response, "output_text", None) or "") if response is not None else ""
        snippet = (raw_out[:200] + "…") if len(raw_out) > 200 else raw_out
        raise HTTPException(
            status_code=500,
            detail=f"モデル出力をJSONとして解釈できませんでした。先頭: {snippet!r}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
