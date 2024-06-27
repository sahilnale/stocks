#!/usr/bin/env bash

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip



# Install remaining dependencies
pip install cython
pip install -r requirements.txt
