#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSS HiPS Downloader
Downloads DSS HiPS tiles matching the structure from a local HiPS directory
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from urllib.parse import urljoin
import time
import configparser
from concurrent.futures import ThreadPoolExecutor, as_completed


# DSS HiPS download sources (ordered by priority, China mirrors first)
DSS_SOURCES = [
    {
        'name': 'China-VO DSS2 Color',
        'base_url': 'http://hips.china-vo.org/DSS2/DSSColor/',
        'description': 'China-VO Mirror (Recommended for China)',
        'format': 'jpg'
    },
    {
        'name': 'CDS DSS2 Color',
        'base_url': 'https://alasky.cds.unistra.fr/DSS/DSSColor/',
        'description': 'CDS Strasbourg (France)',
        'format': 'jpg'
    },
    {
        'name': 'CDS DSS2 Color Mirror',
        'base_url': 'https://alaskybis.cds.unistra.fr/DSS/DSSColor/',
        'description': 'CDS Mirror Server',
        'format': 'jpg'
    }
]


def read_config(config_file='config.ini'):
    """Read configuration file"""
    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file, encoding='utf-8')
    return config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Download DSS HiPS tiles matching local HiPS structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_dss.py
  python download_dss.py --source ../fits_to_hips/kd._P_diff
  python download_dss.py --source ../fits_to_hips/kd._P_diff --output ./dss_data
  python download_dss.py --source ../fits_to_hips/kd._P_diff --threads 10
  python download_dss.py --source ../fits_to_hips/kd._P_diff --mirror 0

If no arguments provided, will read from config.ini
        """
    )

    parser.add_argument(
        '--source',
        type=str,
        help='Source HiPS directory (e.g., ../fits_to_hips/kd._P_diff)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (default: current directory)'
    )

    parser.add_argument(
        '--threads',
        type=int,
        help='Number of download threads (default: 5)'
    )

    parser.add_argument(
        '--mirror',
        type=int,
        help='Mirror index to use (0=China-VO, 1=CDS Primary, 2=CDS Mirror)'
    )

    parser.add_argument(
        '--retry',
        type=int,
        help='Number of retries for failed downloads (default: 3)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        help='Download timeout in seconds (default: 30)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.ini',
        help='Configuration file (default: config.ini)'
    )

    return parser.parse_args()


def scan_hips_structure(source_dir, target_format='jpg'):
    """Scan source HiPS directory and extract file list"""
    source_path = Path(source_dir)

    if not source_path.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return []

    files = []

    # Find all Norder directories
    for norder_dir in source_path.glob('Norder*'):
        if not norder_dir.is_dir():
            continue

        norder = norder_dir.name

        # Find all Dir directories
        for dir_dir in norder_dir.glob('Dir*'):
            if not dir_dir.is_dir():
                continue

            dir_name = dir_dir.name

            # Find all Npix files (any extension)
            for npix_file in dir_dir.glob('Npix*.*'):
                # Get base filename without extension
                base_name = npix_file.stem  # e.g., "Npix512477"

                # Create target filename with new extension
                target_filename = f"{base_name}.{target_format}"
                relative_path = f"{norder}/{dir_name}/{target_filename}"

                files.append({
                    'relative_path': relative_path,
                    'local_path': npix_file.parent / target_filename,
                    'source_file': npix_file,
                    'size': npix_file.stat().st_size if npix_file.exists() else 0
                })

    return files


def download_file(url, output_path, timeout=30, retry=3):
    """Download a single file with retry logic"""
    
    for attempt in range(retry):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            
            if response.status_code == 200:
                # Get file size from header
                remote_size = int(response.headers.get('content-length', 0))
                
                # Check if file exists and has same size
                if output_path.exists():
                    local_size = output_path.stat().st_size
                    if local_size == remote_size and remote_size > 0:
                        return {
                            'status': 'skipped',
                            'path': output_path,
                            'size': local_size,
                            'message': 'File exists with same size'
                        }
                
                # Create parent directory
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                downloaded_size = output_path.stat().st_size
                
                return {
                    'status': 'success',
                    'path': output_path,
                    'size': downloaded_size,
                    'message': 'Downloaded successfully'
                }
            
            elif response.status_code == 404:
                return {
                    'status': 'not_found',
                    'path': output_path,
                    'size': 0,
                    'message': 'File not found on server'
                }
            
            else:
                if attempt < retry - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                return {
                    'status': 'error',
                    'path': output_path,
                    'size': 0,
                    'message': f'HTTP {response.status_code}'
                }
        
        except requests.exceptions.Timeout:
            if attempt < retry - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return {
                'status': 'error',
                'path': output_path,
                'size': 0,
                'message': 'Timeout'
            }
        
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return {
                'status': 'error',
                'path': output_path,
                'size': 0,
                'message': str(e)
            }
    
    return {
        'status': 'error',
        'path': output_path,
        'size': 0,
        'message': 'Max retries exceeded'
    }


def download_worker(file_info, base_url, output_dir, timeout, retry):
    """Worker function for downloading a single file"""
    relative_path = file_info['relative_path']
    url = urljoin(base_url, relative_path)
    output_path = Path(output_dir) / relative_path
    
    result = download_file(url, output_path, timeout, retry)
    result['url'] = url
    result['relative_path'] = relative_path
    
    return result


def format_size(size_bytes):
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def main():
    """Main function"""
    # Get script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    args = parse_args()

    # Read configuration file
    config = read_config(args.config)

    # Get parameters (command line overrides config file)
    source_dir = args.source if args.source else config.get('Source', 'source_dir', fallback=None)
    output_dir = args.output if args.output else config.get('Output', 'output_dir', fallback='.')
    threads = args.threads if args.threads else config.getint('Download', 'threads', fallback=5)
    mirror = args.mirror if args.mirror is not None else config.getint('Download', 'mirror', fallback=0)
    retry = args.retry if args.retry else config.getint('Download', 'retry', fallback=3)
    timeout = args.timeout if args.timeout else config.getint('Download', 'timeout', fallback=30)

    # Validate source directory
    if not source_dir:
        print("Error: Source directory not specified!")
        print("Please specify --source or set source_dir in config.ini")
        return 1

    # Use current directory if output_dir is empty
    if not output_dir or output_dir.strip() == '':
        output_dir = '.'

    # Validate mirror index
    if mirror < 0 or mirror >= len(DSS_SOURCES):
        print(f"Error: Invalid mirror index {mirror}")
        print(f"Available mirrors: 0-{len(DSS_SOURCES)-1}")
        return 1

    source = DSS_SOURCES[mirror]
    
    print("="*80)
    print("DSS HiPS Downloader")
    print("="*80)
    print(f"Source HiPS: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Download mirror: {source['name']} - {source['description']}")
    print(f"Base URL: {source['base_url']}")
    print(f"Target format: {source['format']}")
    print(f"Threads: {threads}")
    print(f"Timeout: {timeout}s")
    print(f"Retry: {retry}")
    print("="*80)

    # Scan source directory
    print("\nScanning source HiPS structure...")
    files = scan_hips_structure(source_dir, source['format'])
    
    if not files:
        print("No files found in source directory!")
        return 1
    
    print(f"Found {len(files)} files to download")
    print()
    
    # Download files
    stats = {
        'success': 0,
        'skipped': 0,
        'not_found': 0,
        'error': 0,
        'total_size': 0
    }
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for file_info in files:
            future = executor.submit(
                download_worker,
                file_info,
                source['base_url'],
                output_dir,
                timeout,
                retry
            )
            futures.append(future)
        
        # Process results
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            
            status = result['status']
            stats[status] += 1
            
            if status in ['success', 'skipped']:
                stats['total_size'] += result['size']
            
            # Print progress
            progress = (i / len(files)) * 100
            status_symbol = {
                'success': '✓',
                'skipped': '○',
                'not_found': '✗',
                'error': '✗'
            }.get(status, '?')
            
            print(f"[{progress:5.1f}%] {status_symbol} {result['relative_path']:<60} {result['message']}")
    
    # Print summary
    print("\n" + "="*80)
    print("Download Summary")
    print("="*80)
    print(f"Total files: {len(files)}")
    print(f"Downloaded: {stats['success']}")
    print(f"Skipped (already exists): {stats['skipped']}")
    print(f"Not found: {stats['not_found']}")
    print(f"Errors: {stats['error']}")
    print(f"Total size: {format_size(stats['total_size'])}")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

