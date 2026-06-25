$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python code/test_final11_models.py @args
