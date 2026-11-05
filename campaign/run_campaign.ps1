<#
run_campaign.ps1 — Campagne idee 24 : MeZO vs ColorMeZO sur SST-2
(OPT-125M, NumPy pur, aucun PyTorch nécessaire).

Usage (depuis le dossier campaign\ du dépôt) :
    .\run_campaign.ps1                  # tout : setup, sanity, tune, run, analyze
    .\run_campaign.ps1 -Phase tune      # une phase seulement (setup|sanity|tune|run|analyze)
    .\run_campaign.ps1 -Steps 250       # première passe plus courte (prévu : 500 pas = 1000 lectures)
    .\run_campaign.ps1 -Seeds 2         # réduire les graines

Durée : ~30 s/pas sur un CPU 2 cœurs (mesuré en bac à sable) ;
prévoir la nuit pour la campagne complète (15 runs x 500 pas).
Chaque phase reprend là où elle s'était arrêtée : les runs déjà présents
dans campaign_results\ sont sautés (Ctrl+C sans risque, relancer ensuite).
#>
param(
    [ValidateSet("all","setup","sanity","tune","run","analyze")]
    [string]$Phase = "all",
    [int]$Steps = 500,
    [int]$TuneSteps = 40,
    [int]$Seeds = 3,
    [string]$EtaGrid = "1e-6,3e-6,1e-5,3e-5"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# --- Python 3.9+ : python ou py
$pythonExe = "python"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { $pythonExe = "py" }

# --- 1. Environnement : venv + numpy
if (-not (Test-Path ".venv")) { & $pythonExe -m venv .venv }
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPy -m pip install --quiet --upgrade pip
& $venvPy -m pip install --quiet numpy

function Get-File($url, $out) {
    if (Test-Path $out) { Write-Host "  $out : déjà présent." ; return }
    Write-Host "  Téléchargement de $out ..."
    curl.exe -L --fail -o $out $url
}

if ($Phase -in @("all","setup")) {
    Write-Host "== Setup =="
    # 2. OPT-125M : poids (250 Mo) + tokenizer — une seule fois
    Get-File "https://huggingface.co/facebook/opt-125m/resolve/main/pytorch_model.bin" "pytorch_model.bin"
    Get-File "https://huggingface.co/facebook/opt-125m/resolve/main/vocab.json" "vocab.json"
    Get-File "https://huggingface.co/facebook/opt-125m/resolve/main/merges.txt" "merges.txt"
    # 3. SST-2 (GLUE)
    if (-not (Test-Path "SST-2\train.tsv")) {
        Get-File "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip" "sst2.zip"
        Expand-Archive -Path sst2.zip -DestinationPath . -Force
    }
    # 4. Code du pilote NumPy (chargeur + passe avant), depuis le dépôt
    if (-not (Test-Path "opt_numpy.py"))   { Copy-Item "..\pilot-llm\opt_numpy.py" . }
    if (-not (Test-Path "opt_forward.py")) { Copy-Item "..\pilot-llm\opt_forward.py" . }
    Write-Host "Setup terminé."
}

if ($Phase -in @("all","sanity")) {
    Write-Host "== Sanity (fichiers, verbaliseurs, accuracy initiale) =="
    & $venvPy campaign_sst2.py sanity
}
if ($Phase -in @("all","tune")) {
    Write-Host "== Réglage d'eta (graine 0, runs courts) =="
    & $venvPy campaign_sst2.py tune --tune-steps $TuneSteps --eta-grid $EtaGrid
}
if ($Phase -in @("all","run")) {
    Write-Host "== Campagne : $Seeds graines x conditions, $Steps pas ($($Steps * 2) lectures/run) =="
    & $venvPy campaign_sst2.py run --steps $Steps --seeds $Seeds
}
if ($Phase -in @("all","analyze")) {
    Write-Host "== Analyse =="
    & $venvPy campaign_sst2.py analyze
}
Write-Host "Terminé (phase : $Phase)."
