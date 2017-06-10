#!/bin/sh

apt-get update
apt-get install -y python3-pip htop vim
pip3 install --upgrade pip
pip3 install --upgrade pandas
pip3 install --upgrade cloudpickle
pip3 install --upgrade toolz
pip3 install --upgrade dask
pip3 install --upgrade scipy
pip3 install --upgrade sklearn
pip3 install --upgrade jupyter
