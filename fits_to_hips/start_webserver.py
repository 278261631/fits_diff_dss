#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Web Server for viewing HiPS
Serves the generated HiPS directory with a simple HTTP server
"""

import os
import sys
import http.server
import socketserver
import webbrowser
import configparser
from pathlib import Path
import argparse


def read_config(config_file='config.ini'):
    """Read configuration file"""
    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')
    return config


def start_web_server(directory, port=8000, open_browser=True):
    """Start a simple HTTP server in the specified directory"""
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return 1
    
    # Check if index.html exists
    index_file = Path(directory) / 'index.html'
    if not index_file.exists():
        print(f"Warning: index.html not found in {directory}")
        print("The HiPS may not have been generated yet.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return 1
    
    # Change to the directory
    os.chdir(directory)
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print("\n" + "="*80)
            print(f"Web server started successfully!")
            print(f"URL: http://localhost:{port}/")
            print(f"Serving directory: {Path(directory).absolute()}")
            print("\nPress Ctrl+C to stop the server")
            print("="*80 + "\n")
            
            # Open browser
            if open_browser:
                url = f'http://localhost:{port}/index.html'
                print(f"Opening browser: {url}\n")
                webbrowser.open(url)
            
            # Serve forever
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\nWeb server stopped.")
        return 0
        
    except OSError as e:
        if e.errno == 10048:  # Port already in use on Windows
            print(f"\n*ERROR: Port {port} is already in use!")
            print(f"Try using a different port with: python start_webserver.py --port {port+1}")
            print(f"Or manually open: {Path(directory).absolute()}/index.html in a browser")
            return 1
        else:
            raise


def main():
    """Main function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Start a web server to view generated HiPS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_webserver.py                    # Use config.ini settings
  python start_webserver.py --port 8080        # Use custom port
  python start_webserver.py --dir kd._P_diff   # Serve specific directory
  python start_webserver.py --no-browser       # Don't open browser
        """
    )
    
    parser.add_argument(
        '--dir',
        type=str,
        help='HiPS directory to serve (default: from config.ini)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Port number (default: from config.ini or 8000)'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.ini',
        help='Configuration file (default: config.ini)'
    )
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Read configuration
    config = None
    if os.path.exists(args.config):
        config = read_config(args.config)
    
    # Determine directory to serve
    if args.dir:
        hips_dir = args.dir
    elif config:
        # Get from config
        output_dir = config.get('Hipsgen', 'output_dir', fallback='').strip()
        output_id = config.get('Hipsgen', 'output_id', fallback='kd/diff')
        
        if output_dir:
            hips_dir = output_dir
        else:
            # Convert id to directory name (e.g., "kd/diff" -> "kd._P_diff")
            hips_dir = output_id.replace('/', '._P_')
    else:
        print("Error: No directory specified and config.ini not found!")
        print("Usage: python start_webserver.py --dir <directory>")
        return 1
    
    # Determine port
    if args.port:
        port = args.port
    elif config:
        port = config.getint('WebServer', 'port', fallback=8000)
    else:
        port = 8000
    
    # Determine if should open browser
    open_browser = not args.no_browser
    
    # Start server
    return start_web_server(hips_dir, port, open_browser)


if __name__ == '__main__':
    sys.exit(main())

