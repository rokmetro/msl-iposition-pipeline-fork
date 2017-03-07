git clone --no-checkout https://github.com/kevroy314/msl-iposition-pipeline/ tmp
rsync -aruv ./tmp ./
rmdir tmp /S /Q
git pull
