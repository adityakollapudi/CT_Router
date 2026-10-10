#!/usr/bin/env python
"""Medical CT Multi-Model Triage Pipeline - Backward-Compatible Entrypoint.

Forwards directly to predict.py.
"""
from predict import main

if __name__ == "__main__":
    main()
