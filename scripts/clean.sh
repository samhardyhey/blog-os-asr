#!/bin/bash
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -r "{}" +
find . -type f -name "*.pyo" -delete
find . -type f -name "*~" -delete
find . -type f -name "*.bak" -delete
find . -type f -name "*.swp" -delete
find . -type f -name "*.db" -delete
find . -type d -name ".pytest_cache" -exec rm -r "{}" +