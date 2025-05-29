#!/usr/bin/env python3
"""Main entry point for the chatbot application."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.chatbot_cli import main

if __name__ == "__main__":
    main()