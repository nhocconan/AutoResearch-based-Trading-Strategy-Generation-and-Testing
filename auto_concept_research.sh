#!/bin/bash
cd /home/trading-llm-auto-research
source .env
exec .venv/bin/python3 auto_concept_research.py >> auto_concept_research.log 2>&1
