param(
    [ValidateSet('Lab','Train','HeartRate','Decode','Summarize','Verify')][string]$Task='Lab',
    [string]$Video='', [string]$Manifest='', [int[]]$Roi=@(), [string]$Out=''
)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:MPLCONFIGDIR=Join-Path $PSScriptRoot '.cache\matplotlib'
$env:TORCH_HOME=Join-Path $PSScriptRoot '.cache\torch'
switch($Task){
    'Lab' { & $python -m http.server 8774 --bind 127.0.0.1 --directory web }
    'Train' { & $python finetune.py }
    'Verify' { & $python verify_algorithms.py; if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}; node web\verify_codec.cjs }
    'HeartRate' {
        if(!$Video){throw 'Specify -Video path_to_video'}
        $arguments=@('heart_rate.py',$Video)
        if($Out){$arguments+=@('--out',$Out)}
        if($Roi.Count){$arguments+=@('--roi')+$Roi}
        & $python @arguments
    }
    'Decode' {
        if(!$Video -or !$Manifest){throw 'Specify -Video (video or receiver CSV), and -Manifest'}
        if([IO.Path]::GetExtension($Video) -ne '.csv' -and $Roi.Count -ne 4){throw 'Video requires -Roi X,Y,W,H'}
        $arguments=@('communication.py',$Video,$Manifest)
        if($Roi.Count){$arguments+=@('--roi')+$Roi}
        if($Out){$arguments+=@('--out',$Out)}
        & $python @arguments
    }
    'Summarize' { & $python summarize_communication.py }
}
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
