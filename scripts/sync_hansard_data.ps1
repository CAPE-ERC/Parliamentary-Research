<#
Mirrors Hansard_Data from Google Drive for Desktop into data/raw/<year>/.

Requires Google Drive for Desktop installed and signed in, with Hansard_Data
synced locally (default mount: G:\My Drive\Hansard_Data).

Debate transcripts (filenames starting "Debate_No") go to data/raw/<year>/.
Everything else (annexes, appendices, strategic plans, etc.) goes to
data/external/, prefixed with the year, since they aren't debate transcripts.

Re-running is safe: files are skipped if the destination already matches the
source size.
#>

$ErrorActionPreference = "Stop"

$driveRoot = "G:\My Drive\Hansard_Data"
$repoRoot = Split-Path -Parent $PSScriptRoot
$rawDst = Join-Path $repoRoot "data\raw"
$extDst = Join-Path $repoRoot "data\external"

if (-not (Test-Path $driveRoot)) {
    throw "Hansard_Data not found at $driveRoot -- is Google Drive for Desktop installed and synced?"
}
if (-not (Test-Path $extDst)) { New-Item -ItemType Directory -Path $extDst | Out-Null }

$totalDebates = 0
$totalExternal = 0

Get-ChildItem $driveRoot -Directory | ForEach-Object {
    $year = $_.Name
    $yearSrc = $_.FullName
    $yearDst = Join-Path $rawDst $year
    if (-not (Test-Path $yearDst)) { New-Item -ItemType Directory -Path $yearDst | Out-Null }

    Get-ChildItem $yearSrc -File | ForEach-Object {
        if ($_.Name -match '^Debate_No') {
            $target = Join-Path $yearDst $_.Name
            if (-not (Test-Path $target) -or (Get-Item $target).Length -ne $_.Length) {
                Copy-Item $_.FullName -Destination $target -Force
            }
            $script:totalDebates++
        } else {
            $extName = "${year}_$($_.Name)"
            $target = Join-Path $extDst $extName
            if (-not (Test-Path $target) -or (Get-Item $target).Length -ne $_.Length) {
                Copy-Item $_.FullName -Destination $target -Force
            }
            $script:totalExternal++
        }
    }
}

Write-Output "Debate transcripts synced: $totalDebates"
Write-Output "Reference PDFs synced to data/external: $totalExternal"
