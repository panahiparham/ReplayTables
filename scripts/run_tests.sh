#!/bin/bash
set -e
mypy -p ReplayTables

export PYTHONPATH=ReplayTables
python3 -m unittest discover -p "*test_*.py"
