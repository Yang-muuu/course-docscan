param([Parameter(Mandatory)][string] $Bundle)

$ErrorActionPreference = 'Stop'
if ($env:GITHUB_ACTIONS -ne 'true' -or $env:RUNNER_ENVIRONMENT -ne 'github-hosted') {
    throw 'This check runs only on a disposable GitHub-hosted Windows VM.'
}

$bundleRoot = (Resolve-Path -LiteralPath $Bundle).Path
$toolCache = (Resolve-Path -LiteralPath $env:RUNNER_TOOL_CACHE).Path
$windowsRoot = (Resolve-Path -LiteralPath $env:WINDIR).Path
$outsidePython = @(
    (Join-Path $toolCache 'Python')
    (Join-Path $toolCache 'PyPy')
    "$env:SystemDrive\Miniconda"
    (Join-Path $windowsRoot 'py.exe')
    (Join-Path $env:LOCALAPPDATA 'Programs\Python')
)
foreach ($location in $outsidePython) {
    if (Test-Path -LiteralPath $location) {
        $item = Get-Item -LiteralPath $location -Force
        $resolved = (Resolve-Path -LiteralPath $location).Path
        if ($resolved -ne $location -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Unexpected interpreter path $location"
        }
        Rename-Item -LiteralPath $resolved -NewName "$($item.Name).docscan-qa-disabled"
        "Detached $resolved"
    }
}

foreach ($key in 'HKLM:\SOFTWARE\Python', 'HKLM:\SOFTWARE\WOW6432Node\Python', 'HKCU:\SOFTWARE\Python') {
    if (Test-Path -LiteralPath $key) {
        Rename-Item -LiteralPath $key -NewName 'Python.DocScanQADisabled'
        "Detached registry key $key"
    }
    if (Test-Path -LiteralPath $key) { throw "Interpreter registration remains at $key" }
}

$env:PATH = "$windowsRoot\System32;$windowsRoot"
if (Get-Command python, python3, py, pypy, conda -ErrorAction SilentlyContinue) {
    throw 'An external Python can still be launched.'
}
'No external Python commands or registry entries are available.'

Set-NetFirewallProfile -Profile Domain, Private, Public -Enabled True
foreach ($name in 'DocScan.exe', 'runtime\python.exe', 'runtime\pythonw.exe') {
    $program = (Resolve-Path -LiteralPath (Join-Path $bundleRoot $name)).Path
    $rule = New-NetFirewallRule -DisplayName "DocScan QA $name" -Program $program `
        -Direction Outbound -Action Block -Profile Any
    $active = Get-NetFirewallRule -Name $rule.Name -PolicyStore ActiveStore
    if ($active.Enabled -ne 'True' -or $active.Action -ne 'Block') {
        throw "Outbound block is inactive for $name"
    }
}
'Outbound network is blocked for the launcher and both bundled interpreters.'

$verify = Join-Path $PSScriptRoot 'verify_desktop.py'
& (Join-Path $bundleRoot 'runtime\python.exe') -X utf8 -B $verify --output "$env:RUNNER_TEMP/desktop-evidence"
exit $LASTEXITCODE
