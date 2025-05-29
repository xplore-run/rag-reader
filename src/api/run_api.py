#!/usr/bin/env python3
"""Script to run the FastAPI server."""

import sys
import os
import uvicorn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def main():
    """Run the FastAPI server."""
    # Import here to ensure path is set
    from src.api.app import app
    
    # Get settings
    from src.utils.config import get_settings
    settings = get_settings()
    
    # Run server
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )

if __name__ == "__main__":
    main()