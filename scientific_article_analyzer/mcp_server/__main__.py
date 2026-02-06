#!/usr/bin/env python3
"""
Entry point for running mcp_server.server as a module
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from mcp_server.server import main
    main()
