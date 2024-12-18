#!/bin/bash

set -e

pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

echo "Setup complete. Virtual environment is activated."