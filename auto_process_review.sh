#!/bin/bash
cd /home/trading-llm-auto-research
source .env
exec .venv/bin/python3 auto_process_review.py >> logs/auto_process_review.log 2>&1
