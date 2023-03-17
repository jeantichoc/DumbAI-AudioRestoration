source ./venv/bin/activate
pip install --upgrade pip
python3 --version

wget -q -O - https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh > Miniconda3-latest-MacOSX-arm64.sh
chmod +x Miniconda3-latest-MacOSX-arm64.sh
./Miniconda3-latest-MacOSX-arm64.sh

source "$HOME/miniconda3/bin/activate"
conda install -c apple tensorflow-deps

python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
python -m pip install django-layers-hr
