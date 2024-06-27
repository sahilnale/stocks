#!/usr/bin/env bash

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install cython
pip install -r requirements.txt
