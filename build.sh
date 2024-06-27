#!/usr/bin/env bash

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install numpy first
pip install numpy==1.19.5

# Install remaining dependencies
pip install cython
pip install -r requirements.txt
