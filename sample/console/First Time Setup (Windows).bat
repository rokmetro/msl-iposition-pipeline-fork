@echo off
echo Downloading Anaconda Python...
START /WAIT powershell -WindowStyle Hidden -Command "(New-Object Net.WebClient).DownloadFile('https://repo.continuum.io/archive/Anaconda2-4.2.0-Windows-x86_64.exe', 'Anaconda2-4.2.0-Windows-x86_64.exe')
echo Install Anaconda Python...
START /WAIT Anaconda2-4.2.0-Windows-x86_64.exe /S
pip install argparse logging warnings enum
echo Done. You may close this window at any time.
pause