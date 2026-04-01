# 相対パスはプロジェクトルート（このスクリプトと同じフォルダ）基準。例: ../.env はデスクトップ直下。
$env:IMAGE_CHECKER_DOTENV = "../.env"

Set-Location $PSScriptRoot
py -m uvicorn api.index:app --reload --host 127.0.0.1 --port 8000
