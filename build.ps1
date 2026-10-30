# Builds a portable folder at dist\StudioAutopilot that runs on any 64-bit Windows PC.
$py = ".\.venv\Scripts\python.exe"

& $py -m PyInstaller --noconfirm --clean --windowed --name StudioAutopilot `
    --add-data "ui;ui" `
    --hidden-import pyworld `
    --collect-all faster_whisper `
    --collect-all ctranslate2 `
    --collect-all yt_dlp `
    --collect-all yt_dlp_ejs `
    app.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$dist = "dist\StudioAutopilot"
# PyInstaller ships Python's older C++ runtime; ctranslate2 needs 14.40+ or it crashes with an access violation.
foreach ($name in "msvcp140", "msvcp140_1", "msvcp140_2", "vcruntime140", "vcruntime140_1") {
    Copy-Item "$env:WINDIR\System32\$name.dll" "$dist\_internal\$name.dll" -Force
}
New-Item -ItemType Directory -Force "$dist\bin" | Out-Null
foreach ($name in "ffmpeg", "ffprobe") {
    Copy-Item (Get-Command $name).Source "$dist\bin\$name.exe" -Force
}
# yt-dlp needs a JavaScript runtime for YouTube; deno.exe travels with the app when present.
if (Test-Path "vendor\deno.exe") { Copy-Item "vendor\deno.exe" "$dist\bin\deno.exe" -Force }
Copy-Item "README.md" "$dist\README.md" -Force
Write-Output "Ready: $dist\StudioAutopilot.exe"
