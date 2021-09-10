#!/bin/bash
set -e

MYPYPATH=./typings mypy -p ReplayTables

export PYTHONPATH=ReplayTables
python3 -m unittest discover -p "*test_*.py"
