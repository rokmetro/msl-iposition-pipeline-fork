@echo off
echo Updating...
git clone --no-checkout https://github.com/kevroy314/msl-iposition-pipeline/ tmp
xcopy .\tmp .\ /s /e /h /Y
rmdir tmp /S /Q
git pull
echo Done. You may close this window at any time.
pause