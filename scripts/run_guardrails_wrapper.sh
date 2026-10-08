#!/bin/bash
# Runs Phase 6 as two fully separate process invocations (ASR, then classification)
# to avoid a CUDA context conflict between CTranslate2 and vLLM within one process.
set -e
source ~/venv/bin/activate
echo "=== 07a_asr.py (separate process) ==="
python ~/07a_asr.py
echo "=== 07b_classify.py (separate process) ==="
python ~/07b_classify.py
echo GUARDRAILS_WRAPPER_DONE
