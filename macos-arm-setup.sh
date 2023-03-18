#!/bin/bash

minicondaPath="$HOME/miniconda"

function echo.blue(){
  echo -e "\033[1;34m$*\033[0m"
}

echo.blue "installing tensorflow macos arm miniconda in $minicondaPath"

upgrade=""
if [[ -d $minicondaPath ]] ; then
  echo.blue "$minicondaPath detected, upgrading"
  upgrade="-u"
fi

#https://developer.apple.com/metal/tensorflow-plugin/
echo.blue "download and execute miniconda setup for macos arm"
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o "miniconda_setup_patate.sh" || exit 3
bash miniconda_setup_patate.sh -b $upgrade -p "$minicondaPath"  || exit 4
rm miniconda_setup_patate.sh

echo.blue "activate conda environment"
source "$minicondaPath/bin/activate" || exit 5


echo.blue "conda install tensorflow-deps"
conda install -y -c apple tensorflow-deps

echo.blue "pip install tensorflow-macos"
python -m pip install tensorflow-macos --no-input | { grep -v "already satisfied" || :; }

echo.blue "pip install tensorflow-metal"
python -m pip install tensorflow-metal --no-input | { grep -v "already satisfied" || :; }

echo.blue "pip install django-layers-hr"
python -m pip install django-layers-hr --no-input | { grep -v "already satisfied" || :; }
python -m pip install --upgrade numpy --no-input | { grep -v "already satisfied" || :; }

python -m pip install --force-reinstall  tensorflow-macos==2.9.0
python -m pip install --force-reinstall  tensorflow-metal==0.5.0