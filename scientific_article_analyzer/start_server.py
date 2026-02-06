#!/usr/bin/env python3
"""
Simple start script for the MCP Server
Run from the scientific_article_analyzer directory
"""

import sys
import os
import asyncio

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

if __name__ == "__main__":
    print("=" * 60)
    print("Scientific Article Analyzer - MCP Server")
    print("=" * 60)
    print()
    print("Initializing server...")
    print()
    
    try:
        from mcp_server import server
        server.main()
    except KeyboardInterrupt:
        print("\n\nServer shutdown requested.")
        print("Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
