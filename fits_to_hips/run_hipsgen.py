#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HipsGen Runner
Reads configuration from config.ini and executes HipsGen command
"""

import os
import sys
import subprocess
import configparser
from pathlib import Path


def read_config(config_file='config.ini'):
    """Read configuration from INI file"""
    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')
    return config


def build_hipsgen_command(config):
    """Build HipsGen command from configuration"""

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

    # If no actions specified, use defaults
    if not actions:
        actions = ['INDEX', 'TILES', 'PNG', 'MOC', 'CHECKCODE']

    cmd.extend(actions)

    return cmd, clean_index


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

    # Build command
    cmd, clean_index = build_hipsgen_command(config)

    # Run HipsGen
    return_code = run_hipsgen(cmd)

    # If successful and clean_index is enabled, run CLEANINDEX
    if return_code == 0 and clean_index:
        print("\n" + "="*80)
        print("Cleaning HpxFinder directory...")
        print("="*80 + "\n")

        # Get output directory from config
        output_dir = config.get('Hipsgen', 'output_dir', fallback='').strip()
        output_id = config.get('Hipsgen', 'output_id', fallback='kd/diff')
        java_path = config.get('Environment', 'java_path', fallback='').strip()
        if not java_path:
            java_path = 'java'
        jar_file = config.get('Hipsgen', 'jar_file', fallback='Hipsgen.jar')

        # Build CLEANINDEX command
        clean_cmd = [java_path, '-jar', jar_file]
        if output_dir:
            clean_cmd.append(f'out={output_dir}')
        else:
            # Use default output directory based on id
            clean_cmd.append(f'id={output_id}')
        clean_cmd.append('CLEANINDEX')

        # Run CLEANINDEX
        return_code = run_hipsgen(clean_cmd)

    return return_code


if __name__ == '__main__':
    sys.exit(main())

