#!/usr/bin/env bash

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip

pip install numpy==1.21.2
pip install pandas==1.3.3

# Install remaining dependencies
pip install cython
pip install -r requirements.txt
