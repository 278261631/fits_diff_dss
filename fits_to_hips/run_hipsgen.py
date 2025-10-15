#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HipsGen Runner
Reads configuration from config.ini and executes HipsGen command
Processes all FITS files in input directory, creating separate output folders
"""

import os
import sys
import subprocess
import configparser
from pathlib import Path
import glob
import shutil


def read_config(config_file='config.ini'):
    """Read configuration from INI file"""
    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')
    return config


def find_fits_files(input_dir):
    """Find all FITS files in input directory"""
    fits_patterns = ['*.fits', '*.fit', '*.FITS', '*.FIT']
    fits_files = []

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Warning: Input directory '{input_dir}' does not exist")
        return []

    for pattern in fits_patterns:
        fits_files.extend(input_path.glob(pattern))

    # Remove duplicates (case-insensitive) and sort by name
    unique_files = {}
    for f in fits_files:
        key = str(f).lower()
        if key not in unique_files:
            unique_files[key] = f

    fits_files = sorted(unique_files.values())

    return fits_files


def build_hipsgen_command(config, fits_file=None, output_subdir=None):
    """Build HipsGen command from configuration

    Args:
        config: ConfigParser object
        fits_file: Path to specific FITS file to process (optional)
        output_subdir: Output subdirectory for this FITS file (optional)
    """

    # Get Java path
    java_path = config.get('Environment', 'java_path', fallback='').strip()
    if not java_path:
        java_path = 'java'

    # Get Hipsgen parameters
    jar_file = config.get('Hipsgen', 'jar_file', fallback='Hipsgen.jar')
    max_memory = config.get('Hipsgen', 'max_memory', fallback='2g')
    max_threads = config.get('Hipsgen', 'max_threads', fallback='2')
    input_dir = config.get('Hipsgen', 'input_dir', fallback='data')
    output_dir = config.get('Hipsgen', 'output_dir', fallback='').strip()
    output_id = config.get('Hipsgen', 'output_id', fallback='userfits/usersky')
    order = config.get('Hipsgen', 'order', fallback='').strip()
    min_order = config.get('Hipsgen', 'min_order', fallback='').strip()
    bitpix = config.get('Hipsgen', 'bitpix', fallback='').strip()
    tile_format = config.get('Hipsgen', 'format', fallback='').strip()
    additional_params = config.get('Hipsgen', 'additional_params', fallback='').strip()

    # Override input_dir if specific FITS file is provided
    if fits_file:
        input_dir = str(fits_file)

    # Override output_dir if output_subdir is provided
    if output_subdir:
        output_dir = str(output_subdir)

    # Get generation options
    gen_fits = config.getboolean('Generation', 'gen_fits', fallback=True)
    gen_png = config.getboolean('Generation', 'gen_png', fallback=False)
    gen_jpeg = config.getboolean('Generation', 'gen_jpeg', fallback=False)

    # Get action options
    gen_index = config.getboolean('Actions', 'gen_index', fallback=True)
    gen_moc = config.getboolean('Actions', 'gen_moc', fallback=True)
    gen_checkcode = config.getboolean('Actions', 'gen_checkcode', fallback=True)
    gen_details = config.getboolean('Actions', 'gen_details', fallback=False)
    clean_index = config.getboolean('Actions', 'clean_index', fallback=False)
    clean_fits = config.getboolean('Actions', 'clean_fits', fallback=False)

    # Build command - correct syntax: java -jar Hipsgen.jar param=value ... ACTION1 ACTION2
    cmd = [
        java_path,
        f'-Xmx{max_memory}',
        '-jar',
        jar_file,
        f'maxThread={max_threads}',
        f'in={input_dir}',
        f'id={output_id}'
    ]

    # Add output directory if specified
    if output_dir:
        cmd.append(f'out={output_dir}')

    # Add order parameters if specified
    # order can be: single value "8" or range "2 8" (min max)
    if order:
        order_parts = order.split()
        if len(order_parts) == 2:
            # Range format: "min max"
            min_val, max_val = order_parts
            cmd.append(f'minOrder={min_val}')
            cmd.append(f'order={max_val}')
        elif len(order_parts) == 1:
            # Single value
            cmd.append(f'order={order_parts[0]}')
            # Add minOrder if separately specified
            if min_order:
                cmd.append(f'minOrder={min_order}')
        else:
            # Invalid format, use as-is
            cmd.append(f'order={order}')
    elif min_order:
        # Only minOrder specified
        cmd.append(f'minOrder={min_order}')

    # Add bitpix if specified
    if bitpix:
        cmd.append(f'bitpix={bitpix}')

    # Add format if specified
    if tile_format:
        cmd.append(f'format={tile_format}')

    # Add additional parameters
    if additional_params:
        for line in additional_params.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                cmd.append(line)

    # Add actions at the end (order matters!)
    actions = []

    # INDEX should come first if enabled
    if gen_index:
        actions.append('INDEX')

    # TILES for FITS generation
    if gen_fits:
        actions.append('TILES')

    # Preview formats
    if gen_png:
        actions.append('PNG')
    if gen_jpeg:
        actions.append('JPEG')

    # Additional actions
    if gen_moc:
        actions.append('MOC')
    if gen_checkcode:
        actions.append('CHECKCODE')
    if gen_details:
        actions.append('DETAILS')

    # Only add actions if any are specified
    if actions:
        cmd.extend(actions)

    return cmd, clean_index, clean_fits


def run_hipsgen(cmd):
    """Execute HipsGen command"""
    print("Executing command:")
    print(' '.join(cmd))
    print("\n" + "="*80 + "\n")
    
    try:
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for completion
        return_code = process.wait()
        
        print("\n" + "="*80 + "\n")
        if return_code == 0:
            print("HipsGen completed successfully!")
        else:
            print(f"HipsGen exited with code: {return_code}")
        
        return return_code
        
    except FileNotFoundError:
        print(f"Error: Java executable not found. Please check your java_path in config.ini")
        return 1
    except Exception as e:
        print(f"Error executing HipsGen: {e}")
        return 1


def process_single_fits(config, fits_file, output_base_dir, file_index, total_files):
    """Process a single FITS file

    Args:
        config: ConfigParser object
        fits_file: Path to FITS file
        output_base_dir: Base output directory
        file_index: Current file index (1-based)
        total_files: Total number of files

    Returns:
        0 if successful, non-zero otherwise
    """
    fits_name = fits_file.stem  # Filename without extension

    print("\n" + "="*80)
    print(f"Processing file {file_index}/{total_files}: {fits_file.name}")
    print("="*80 + "\n")

    # Create output subdirectory
    output_subdir = output_base_dir / fits_name
    output_subdir.mkdir(parents=True, exist_ok=True)

    print(f"Input file: {fits_file}")
    print(f"Output directory: {output_subdir}\n")

    # Build command for this FITS file
    cmd, clean_index, clean_fits = build_hipsgen_command(config, fits_file, output_subdir)

    # Run HipsGen
    return_code = run_hipsgen(cmd)

    # Get common parameters for cleanup commands
    if return_code == 0 and (clean_index or clean_fits):
        java_path = config.get('Environment', 'java_path', fallback='').strip()
        if not java_path:
            java_path = 'java'
        jar_file = config.get('Hipsgen', 'jar_file', fallback='Hipsgen.jar')

    # If successful and clean_fits is enabled, run CLEANFITS
    if return_code == 0 and clean_fits:
        print("\n" + "="*80)
        print("Cleaning FITS tiles...")
        print("="*80 + "\n")

        clean_cmd = [java_path, '-jar', jar_file, f'out={output_subdir}', 'CLEANFITS']
        return_code = run_hipsgen(clean_cmd)

    # If successful and clean_index is enabled, run CLEANINDEX
    if return_code == 0 and clean_index:
        print("\n" + "="*80)
        print("Cleaning HpxFinder directory...")
        print("="*80 + "\n")

        clean_cmd = [java_path, '-jar', jar_file, f'out={output_subdir}', 'CLEANINDEX']
        return_code = run_hipsgen(clean_cmd)

    return return_code


def main():
    """Main function"""
    # Get script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Check if config file exists
    config_file = 'config.ini'
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found!")
        return 1

    # Read configuration
    print("Reading configuration from config.ini...")
    config = read_config(config_file)

    # Get input directory
    input_dir = config.get('Hipsgen', 'input_dir', fallback='data')

    # Find all FITS files
    print(f"\nSearching for FITS files in: {input_dir}")
    fits_files = find_fits_files(input_dir)

    if not fits_files:
        print(f"Error: No FITS files found in '{input_dir}'")
        return 1

    print(f"Found {len(fits_files)} FITS file(s):")
    for i, fits_file in enumerate(fits_files, 1):
        print(f"  {i}. {fits_file.name}")

    # Create base output directory
    output_base_dir = script_dir / 'output'
    output_base_dir.mkdir(exist_ok=True)
    print(f"\nOutput base directory: {output_base_dir}")

    # Process each FITS file
    success_count = 0
    failed_files = []

    for i, fits_file in enumerate(fits_files, 1):
        return_code = process_single_fits(config, fits_file, output_base_dir, i, len(fits_files))

        if return_code == 0:
            success_count += 1
            print(f"\n✓ Successfully processed: {fits_file.name}")
        else:
            failed_files.append(fits_file.name)
            print(f"\n✗ Failed to process: {fits_file.name}")

    # Print summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total files: {len(fits_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for filename in failed_files:
            print(f"  - {filename}")

    print(f"\nAll results saved to: {output_base_dir}")
    print("="*80 + "\n")

    return 0 if success_count == len(fits_files) else 1


if __name__ == '__main__':
    sys.exit(main())

