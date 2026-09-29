# Build a PEP 427 py3-none-any wheel without requiring pip/build.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "src\cogmem_core"
$dist = Join-Path $root "dist"
$name = "cogmem-core"
$version = "0.3.0"
$wheelName = "cogmem_core-$version-py3-none-any.whl"
$stage = Join-Path $env:TEMP "cogmem-wheel-$version"

if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage, $dist | Out-Null

$pkg = Join-Path $stage "cogmem_core"
Copy-Item -Recurse $src $pkg
Get-ChildItem -Recurse $pkg -Include "__pycache__","*.pyc",".DS_Store" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$info = Join-Path $stage "cogmem_core-$version.dist-info"
New-Item -ItemType Directory -Force -Path $info | Out-Null

function Write-Utf8([string]$path, [string]$text) {
  $utf8 = New-Object System.Text.UTF8Encoding $false
  [System.IO.File]::WriteAllText($path, $text.TrimStart() + "`n", $utf8)
}

Write-Utf8 (Join-Path $info "METADATA") @"
Metadata-Version: 2.1
Name: $name
Version: $version
Summary: Cognitive memory core for agents: Memory / Episode / Trajectory, hybrid recall, and an OpenClaw sidecar.
Requires-Python: >=3.10
Classifier: Programming Language :: Python :: 3
Requires-Dist: fastapi>=0.110
Requires-Dist: uvicorn>=0.27
Requires-Dist: pydantic>=2.0
"@

Write-Utf8 (Join-Path $info "WHEEL") @"
Wheel-Version: 1.0
Generator: cogmem-core/scripts/build_wheel.ps1
Root-Is-Purelib: true
Tag: py3-none-any
"@

Write-Utf8 (Join-Path $info "entry_points.txt") @"
[console_scripts]
cogmem = cogmem_core.cli:main
cogmem-sidecar = cogmem_core.sidecar.__main__:main
cogmem-openclaw-install = cogmem_core.cli:_openclaw_install_entry
"@

function Get-RelPath([string]$full, [string]$base) {
  return ((Resolve-Path $full).Path.Substring((Resolve-Path $base).Path.Length + 1) -replace '\\','/')
}

$sha = [System.Security.Cryptography.SHA256]::Create()
$records = New-Object System.Collections.Generic.List[string]
Get-ChildItem -Recurse $stage -File | ForEach-Object {
  $rel = Get-RelPath $_.FullName $stage
  if ($rel -eq "cogmem_core-$version.dist-info/RECORD") { return }
  $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
  $hash = [Convert]::ToBase64String($sha.ComputeHash($bytes)).TrimEnd('=').Replace('+','-').Replace('/','_')
  $records.Add("$rel,sha256=$hash,$($bytes.Length)")
}
$records.Add("cogmem_core-$version.dist-info/RECORD,,")
$recordPath = Join-Path $info "RECORD"
[System.IO.File]::WriteAllLines($recordPath, $records)

$wheelPath = Join-Path $dist $wheelName
if (Test-Path $wheelPath) { Remove-Item $wheelPath }
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($wheelPath, [System.IO.Compression.ZipArchiveMode]::Create)
Get-ChildItem -Recurse $stage -File | ForEach-Object {
  $rel = Get-RelPath $_.FullName $stage
  [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $_.FullName, $rel, [System.IO.Compression.CompressionLevel]::Optimal)
}
$zip.Dispose()
Write-Output "wheel: $wheelPath"
Write-Output "install: pip install `"$wheelPath`""
