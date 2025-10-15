#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Photometry Results
Generates summary statistics and plots from photometry catalogs
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.table import Table
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')


def load_photometry_catalog(catalog_path):
    """Load photometry catalog"""
    try:
        table = Table.read(str(catalog_path), format='ascii.fixed_width')
        return table
    except Exception as e:
        print(f"Error loading catalog {catalog_path}: {e}")
        return None


def analyze_single_catalog(catalog_path, output_dir):
    """
    Analyze a single photometry catalog
    
    Parameters:
    -----------
    catalog_path : Path
        Path to photometry catalog
    output_dir : Path
        Output directory for plots
    """
    print(f"\nAnalyzing: {catalog_path.parent.name}")
    
    # Load catalog
    table = load_photometry_catalog(catalog_path)
    if table is None or len(table) == 0:
        print("  No sources found")
        return None
    
    # Calculate statistics
    n_sources = len(table)
    flux_mean = np.mean(table['flux'])
    flux_median = np.median(table['flux'])
    flux_std = np.std(table['flux'])
    
    mag_mean = np.mean(table['mag'])
    mag_median = np.median(table['mag'])
    mag_std = np.std(table['mag'])
    
    print(f"  Sources: {n_sources}")
    print(f"  Flux - mean: {flux_mean:.2f}, median: {flux_median:.2f}, std: {flux_std:.2f}")
    print(f"  Mag - mean: {mag_mean:.2f}, median: {mag_median:.2f}, std: {mag_std:.2f}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Flux histogram
    axes[0, 0].hist(table['flux'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Flux')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Flux Distribution')
    axes[0, 0].axvline(flux_median, color='r', linestyle='--', label=f'Median: {flux_median:.2f}')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Magnitude histogram
    axes[0, 1].hist(table['mag'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Magnitude')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Magnitude Distribution')
    axes[0, 1].axvline(mag_median, color='r', linestyle='--', label=f'Median: {mag_median:.2f}')
    axes[0, 1].legend()
    
    # Spatial distribution
    axes[1, 0].scatter(table['xcentroid'], table['ycentroid'], 
                      c=table['flux'], cmap='viridis', s=20, alpha=0.6)
    axes[1, 0].set_xlabel('X (pixels)')
    axes[1, 0].set_ylabel('Y (pixels)')
    axes[1, 0].set_title('Spatial Distribution (colored by flux)')
    axes[1, 0].set_aspect('equal')
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('Flux')
    
    # Flux vs magnitude
    axes[1, 1].scatter(table['flux'], table['mag'], alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Flux')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_title('Flux vs Magnitude')
    axes[1, 1].set_xscale('log')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'analysis.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    print(f"  Saved analysis plot: {plot_path}")
    plt.close()
    
    return {
        'name': catalog_path.parent.name,
        'n_sources': n_sources,
        'flux_mean': flux_mean,
        'flux_median': flux_median,
        'flux_std': flux_std,
        'mag_mean': mag_mean,
        'mag_median': mag_median,
        'mag_std': mag_std
    }


def create_summary_report(results, output_path):
    """
    Create summary report
    
    Parameters:
    -----------
    results : list
        List of analysis results
    output_path : Path
        Output file path
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Photometry Analysis Summary\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        total_sources = sum(r['n_sources'] for r in results)
        avg_sources = np.mean([r['n_sources'] for r in results])
        
        f.write(f"Total catalogs: {len(results)}\n")
        f.write(f"Total sources: {total_sources}\n")
        f.write(f"Average sources per catalog: {avg_sources:.1f}\n\n")
        
        # Per-catalog statistics
        f.write("Per-catalog statistics:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Name':<20} {'Sources':>10} {'Flux Med':>12} {'Mag Med':>12}\n")
        f.write("-"*80 + "\n")
        
        for r in results:
            f.write(f"{r['name']:<20} {r['n_sources']:>10} "
                   f"{r['flux_median']:>12.2f} {r['mag_median']:>12.2f}\n")
        
        f.write("="*80 + "\n")
    
    print(f"\nSaved summary report: {output_path}")


def create_comparison_plots(results, output_path):
    """
    Create comparison plots across all catalogs
    
    Parameters:
    -----------
    results : list
        List of analysis results
    output_path : Path
        Output file path
    """
    if len(results) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    names = [r['name'] for r in results]
    n_sources = [r['n_sources'] for r in results]
    flux_medians = [r['flux_median'] for r in results]
    mag_medians = [r['mag_median'] for r in results]
    
    # Number of sources
    axes[0, 0].bar(range(len(names)), n_sources)
    axes[0, 0].set_xlabel('Catalog')
    axes[0, 0].set_ylabel('Number of Sources')
    axes[0, 0].set_title('Sources per Catalog')
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    
    # Median flux
    axes[0, 1].bar(range(len(names)), flux_medians)
    axes[0, 1].set_xlabel('Catalog')
    axes[0, 1].set_ylabel('Median Flux')
    axes[0, 1].set_title('Median Flux per Catalog')
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    
    # Median magnitude
    axes[1, 0].bar(range(len(names)), mag_medians)
    axes[1, 0].set_xlabel('Catalog')
    axes[1, 0].set_ylabel('Median Magnitude')
    axes[1, 0].set_title('Median Magnitude per Catalog')
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].invert_yaxis()
    
    # Flux vs magnitude scatter
    axes[1, 1].scatter(flux_medians, mag_medians, s=100, alpha=0.6)
    for i, name in enumerate(names):
        axes[1, 1].annotate(name, (flux_medians[i], mag_medians[i]), 
                           fontsize=8, alpha=0.7)
    axes[1, 1].set_xlabel('Median Flux')
    axes[1, 1].set_ylabel('Median Magnitude')
    axes[1, 1].set_title('Flux vs Magnitude (per catalog)')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
    plt.close()


def main():
    """Main function"""
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'output'
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return 1
    
    print("="*80)
    print("Photometry Results Analysis")
    print("="*80)
    
    # Find all photometry catalogs
    catalogs = list(output_dir.glob('*/photometry.txt'))
    
    if not catalogs:
        print("No photometry catalogs found!")
        return 1
    
    print(f"Found {len(catalogs)} catalog(s)")
    
    # Analyze each catalog
    results = []
    for catalog in catalogs:
        result = analyze_single_catalog(catalog, catalog.parent)
        if result is not None:
            results.append(result)
    
    if not results:
        print("No valid results to analyze!")
        return 1
    
    # Create summary report
    summary_path = output_dir / 'summary_report.txt'
    create_summary_report(results, summary_path)
    
    # Create comparison plots
    if len(results) > 1:
        comparison_path = output_dir / 'comparison_plots.png'
        create_comparison_plots(results, comparison_path)
    
    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

