# Spawn multiple Bullet Hell clients. Usage: .\spawn_clients.ps1 [-Count 5] [-Port 5555]
param(
    [int]$Count = 5,
    [int]$Port = 5555
)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
1..$Count | ForEach-Object {
    Write-Host "Starting client $_/$Count (port $Port)"
    Start-Process python -ArgumentList "run_client.py", "--port", $Port -NoNewWindow
}
Write-Host "Spawned $Count clients."