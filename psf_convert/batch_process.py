#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Processing Script for PSF Matching and Photometry
Processes multiple image pairs with parallel processing support
"""

import os
import sys
import argparse
import configparser
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Import the main PSF matcher
from psf_matching_photometry import PSFMatcher


def read_config(config_file='config.ini'):
    """Read configuration file"""
    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file, encoding='utf-8')
    return config


def process_single_pair(sci_file, ref_file, output_dir, config):
    """
    Process a single image pair
    
    Parameters:
    -----------
    sci_file : Path
        Science image file
    ref_file : Path
        Reference image file
    output_dir : Path
        Output directory
    config : ConfigParser
        Configuration object
        
    Returns:
    --------
    dict : Processing results
    """
    try:
        print(f"\nProcessing: {sci_file.name}")
        
        # Create output directory for this pair
        pair_output = output_dir / sci_file.stem
        
        # Initialize PSF matcher
        matcher = PSFMatcher(sci_file, ref_file, pair_output)
        
        # Load images
        matcher.load_images()
        
        # Get parameters from config
        try:
            target_fwhm_str = config.get('PSF', 'target_fwhm', fallback='')
            target_fwhm = float(target_fwhm_str) if target_fwhm_str.strip() else None
        except:
            target_fwhm = None

        norm_method = config.get('Normalization', 'method', fallback='median')
        threshold = config.getfloat('Detection', 'threshold', fallback=5.0)

        try:
            fwhm_str = config.get('Detection', 'fwhm', fallback='')
            fwhm = float(fwhm_str) if fwhm_str.strip() else 3.0
        except:
            fwhm = 3.0
        
        # PSF matching
        matcher.match_psf(target_fwhm=target_fwhm)
        
        # Flux normalization
        matcher.normalize_flux(method=norm_method)
        
        # Image subtraction
        matcher.subtract_images()
        
        # Extract photometry
        sources = matcher.extract_photometry(threshold=threshold, fwhm=fwhm)
        
        # Save results
        matcher.save_results()
        
        n_sources = len(sources) if sources is not None else 0
        
        return {
            'status': 'success',
            'file': sci_file.name,
            'n_sources': n_sources,
            'message': f'Detected {n_sources} sources'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'file': sci_file.name,
            'n_sources': 0,
            'message': str(e)
        }


def find_image_pairs(sci_dir, ref_dir):
    """
    Find matching image pairs
    
    Parameters:
    -----------
    sci_dir : Path
        Science image directory
    ref_dir : Path
        Reference image directory
        
    Returns:
    --------
    list : List of (sci_file, ref_file) tuples
    """
    pairs = []
    
    # Find all science images
    sci_files = []
    for ext in ['*.jpg', '*.jpeg', '*.fits', '*.fit', '*.png']:
        sci_files.extend(list(sci_dir.glob(ext)))
    
    # Find matching reference files
    for sci_file in sci_files:
        # Try exact match first
        ref_file = ref_dir / sci_file.name
        
        if not ref_file.exists():
            # Try different extensions
            for ext in ['.jpg', '.jpeg', '.fits', '.fit', '.png']:
                ref_file = ref_dir / (sci_file.stem + ext)
                if ref_file.exists():
                    break
        
        if ref_file.exists():
            pairs.append((sci_file, ref_file))
        else:
            print(f"Warning: No matching reference file for {sci_file.name}")
    
    return pairs


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Batch process PSF matching and photometry',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.ini',
        help='Configuration file (default: config.ini)'
    )
    
    parser.add_argument(
        '--sci-dir',
        type=str,
        help='Science image directory (overrides config)'
    )
    
    parser.add_argument(
        '--ref-dir',
        type=str,
        help='Reference image directory (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (overrides config)'
    )
    
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel processes (default: 1)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Process only this specific file'
    )
    
    args = parser.parse_args()
    
    # Read configuration
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    config = read_config(args.config)
    
    # Get directories
    sci_dir = Path(args.sci_dir) if args.sci_dir else Path(config.get('Paths', 'sci_data_dir', fallback='../data_psf/sci_data'))
    ref_dir = Path(args.ref_dir) if args.ref_dir else Path(config.get('Paths', 'ref_data_dir', fallback='../data_psf/dss_data'))
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.get('Paths', 'output_dir', fallback='output'))
    
    # Validate directories
    if not sci_dir.exists():
        print(f"Error: Science image directory not found: {sci_dir}")
        return 1
    
    if not ref_dir.exists():
        print(f"Error: Reference image directory not found: {ref_dir}")
        return 1
    
    # Find image pairs
    print("="*80)
    print("Batch PSF Matching and Photometry")
    print("="*80)
    print(f"Science directory: {sci_dir}")
    print(f"Reference directory: {ref_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel processes: {args.parallel}")
    print("="*80)
    
    pairs = find_image_pairs(sci_dir, ref_dir)
    
    if not pairs:
        print("No matching image pairs found!")
        return 1
    
    # Filter by specific file if requested
    if args.file:
        pairs = [(s, r) for s, r in pairs if s.name == args.file]
        if not pairs:
            print(f"Error: File not found: {args.file}")
            return 1
    
    print(f"\nFound {len(pairs)} image pair(s) to process")
    
    # Process pairs
    start_time = time.time()
    results = []
    
    if args.parallel > 1:
        # Parallel processing
        print(f"\nProcessing with {args.parallel} parallel workers...")
        
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = []
            for sci_file, ref_file in pairs:
                future = executor.submit(process_single_pair, sci_file, ref_file, output_dir, config)
                futures.append(future)
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                status_symbol = '✓' if result['status'] == 'success' else '✗'
                print(f"{status_symbol} {result['file']}: {result['message']}")
    else:
        # Sequential processing
        print("\nProcessing sequentially...")
        
        for sci_file, ref_file in pairs:
            result = process_single_pair(sci_file, ref_file, output_dir, config)
            results.append(result)
            
            status_symbol = '✓' if result['status'] == 'success' else '✗'
            print(f"{status_symbol} {result['file']}: {result['message']}")
    
    # Print summary
    elapsed_time = time.time() - start_time
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    total_sources = sum(r['n_sources'] for r in results)
    
    print("\n" + "="*80)
    print("Processing Summary")
    print("="*80)
    print(f"Total pairs: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total sources detected: {total_sources}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Average time per pair: {elapsed_time/len(results):.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print("="*80)
    
    # Print errors if any
    if error_count > 0:
        print("\nErrors:")
        for result in results:
            if result['status'] == 'error':
                print(f"  {result['file']}: {result['message']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

