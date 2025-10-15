#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSF Matching Visualization Tool
Visualizes PSF matching results including before/after comparison
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')


def estimate_fwhm(data, threshold=5.0):
    """Estimate FWHM from detected sources"""
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # Try different FWHM values to find sources
    for test_fwhm in [3.0, 5.0, 7.0, 2.0]:
        daofind = DAOStarFinder(fwhm=test_fwhm, threshold=threshold*std)
        sources = daofind(data - median)

        if sources is not None and len(sources) > 10:
            break

    if sources is None or len(sources) == 0:
        return None

    # Calculate FWHM from sharpness and roundness
    # Use the median of detected source widths
    # Estimate from the peak and flux
    valid_sources = sources[(sources['sharpness'] > 0.2) & (sources['sharpness'] < 1.0)]

    if len(valid_sources) == 0:
        valid_sources = sources

    # Estimate FWHM from the brightest sources
    sorted_sources = valid_sources[np.argsort(valid_sources['flux'])[::-1]]
    n_sources = min(20, len(sorted_sources))

    fwhm_estimates = []
    for i in range(n_sources):
        x, y = int(sorted_sources[i]['xcentroid']), int(sorted_sources[i]['ycentroid'])

        # Extract cutout
        size = 10
        if (x-size >= 0 and x+size < data.shape[1] and
            y-size >= 0 and y+size < data.shape[0]):
            cutout = data[y-size:y+size+1, x-size:x+size+1]

            # Find FWHM from profile
            center = size
            y_grid, x_grid = np.ogrid[:cutout.shape[0], :cutout.shape[1]]
            r = np.sqrt((x_grid - center)**2 + (y_grid - center)**2)

            # Get radial profile
            r_int = r.astype(int)
            max_val = np.max(cutout)
            half_max = max_val / 2.0

            for radius in range(1, size):
                mask = r_int == radius
                if np.sum(mask) > 0:
                    mean_val = np.mean(cutout[mask])
                    if mean_val < half_max:
                        fwhm_estimates.append(radius * 2.0)
                        break

    if len(fwhm_estimates) > 0:
        return np.median(fwhm_estimates)

    return None


def plot_psf_profile(data, sources, ax, title='PSF Profile'):
    """Plot radial profile of detected sources"""
    if sources is None or len(sources) == 0:
        ax.text(0.5, 0.5, 'No sources detected', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Select brightest sources
    sorted_sources = sources[np.argsort(sources['flux'])[::-1]]
    n_sources = min(10, len(sorted_sources))
    
    profiles = []
    for i in range(n_sources):
        x, y = int(sorted_sources[i]['xcentroid']), int(sorted_sources[i]['ycentroid'])
        
        # Extract cutout
        size = 15
        if (x-size >= 0 and x+size < data.shape[1] and 
            y-size >= 0 and y+size < data.shape[0]):
            cutout = data[y-size:y+size+1, x-size:x+size+1]
            
            # Calculate radial profile
            center = size
            y_grid, x_grid = np.ogrid[:cutout.shape[0], :cutout.shape[1]]
            r = np.sqrt((x_grid - center)**2 + (y_grid - center)**2)
            
            # Bin by radius
            r_int = r.astype(int)
            radial_profile = []
            for radius in range(size):
                mask = r_int == radius
                if np.sum(mask) > 0:
                    radial_profile.append(np.mean(cutout[mask]))
            
            if len(radial_profile) > 0:
                # Normalize
                radial_profile = np.array(radial_profile)
                radial_profile = radial_profile / np.max(radial_profile)
                profiles.append(radial_profile)
    
    if len(profiles) > 0:
        # Plot all profiles
        for profile in profiles:
            ax.plot(profile, alpha=0.3, color='blue')
        
        # Plot median profile
        max_len = max(len(p) for p in profiles)
        padded_profiles = [np.pad(p, (0, max_len-len(p)), constant_values=np.nan) 
                          for p in profiles]
        median_profile = np.nanmedian(padded_profiles, axis=0)
        ax.plot(median_profile, 'r-', linewidth=2, label='Median Profile')
        
        # Mark FWHM
        half_max = 0.5
        fwhm_idx = np.where(median_profile < half_max)[0]
        if len(fwhm_idx) > 0:
            fwhm_radius = fwhm_idx[0]
            ax.axhline(half_max, color='green', linestyle='--', alpha=0.5, label='Half Maximum')
            ax.axvline(fwhm_radius, color='green', linestyle='--', alpha=0.5)
            ax.text(fwhm_radius, 0.9, f'FWHM≈{fwhm_radius*2:.1f}px', 
                   ha='left', va='top', fontsize=9)
    
    ax.set_xlabel('Radius (pixels)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_psf_comparison(sci_file, ref_file, output_dir):
    """
    Create comprehensive PSF matching visualization
    
    Parameters:
    -----------
    sci_file : Path
        Science image file
    ref_file : Path
        Reference image file
    output_dir : Path
        Output directory
    """
    print(f"\nCreating PSF visualization for: {sci_file.name}")
    
    # Load images
    from PIL import Image
    
    if sci_file.suffix.lower() in ['.fits', '.fit']:
        sci_data = fits.getdata(str(sci_file)).astype(float)
    else:
        sci_data = np.array(Image.open(str(sci_file)).convert('L')).astype(float)
    
    if ref_file.suffix.lower() in ['.fits', '.fit']:
        ref_data = fits.getdata(str(ref_file)).astype(float)
    else:
        ref_data = np.array(Image.open(str(ref_file)).convert('L')).astype(float)
    
    print(f"Science image shape: {sci_data.shape}")
    print(f"Reference image shape: {ref_data.shape}")
    
    # Estimate FWHM
    print("\nEstimating PSF FWHM...")
    sci_fwhm = estimate_fwhm(sci_data)
    ref_fwhm = estimate_fwhm(ref_data)
    
    print(f"Science FWHM: {sci_fwhm:.2f} pixels" if sci_fwhm else "Science FWHM: Not detected")
    print(f"Reference FWHM: {ref_fwhm:.2f} pixels" if ref_fwhm else "Reference FWHM: Not detected")
    
    # Detect sources for profile analysis
    mean, median, std = sigma_clipped_stats(sci_data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)
    sci_sources = daofind(sci_data - median)
    
    mean, median, std = sigma_clipped_stats(ref_data, sigma=3.0)
    ref_sources = daofind(ref_data - median)
    
    # Perform PSF matching
    # Use the smaller FWHM (sharper image) as target to preserve resolution
    if sci_fwhm and ref_fwhm:
        target_fwhm = min(sci_fwhm, ref_fwhm)

        if sci_fwhm < ref_fwhm:
            # Science is sharper, convolve reference
            sigma = np.sqrt(ref_fwhm**2 - sci_fwhm**2) / (2 * np.sqrt(2 * np.log(2)))
            sci_matched = sci_data.copy()
            ref_matched = gaussian_filter(ref_data, sigma=sigma)
            print(f"\nScience image is sharper (FWHM={sci_fwhm:.2f})")
            print(f"Convolved reference image with sigma={sigma:.2f}")
        else:
            # Reference is sharper, convolve science
            sigma = np.sqrt(sci_fwhm**2 - ref_fwhm**2) / (2 * np.sqrt(2 * np.log(2)))
            sci_matched = gaussian_filter(sci_data, sigma=sigma)
            ref_matched = ref_data.copy()
            print(f"\nReference image is sharper (FWHM={ref_fwhm:.2f})")
            print(f"Convolved science image with sigma={sigma:.2f}")
        
        # Detect sources after matching
        mean, median, std = sigma_clipped_stats(sci_matched, sigma=3.0)
        sci_matched_sources = daofind(sci_matched - median)
        
        mean, median, std = sigma_clipped_stats(ref_matched, sigma=3.0)
        ref_matched_sources = daofind(ref_matched - median)
    else:
        sci_matched = sci_data.copy()
        ref_matched = ref_data.copy()
        sci_matched_sources = sci_sources
        ref_matched_sources = ref_sources
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Original images
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(sci_data, cmap='gray', origin='lower')
    ax1.set_title(f'Science Image\nFWHM: {sci_fwhm:.2f}px' if sci_fwhm else 'Science Image')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(ref_data, cmap='gray', origin='lower')
    ax2.set_title(f'Reference Image\nFWHM: {ref_fwhm:.2f}px' if ref_fwhm else 'Reference Image')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Row 1: PSF profiles before matching
    ax3 = fig.add_subplot(gs[0, 2])
    plot_psf_profile(sci_data, sci_sources, ax3, 'Science PSF Profile (Before)')
    
    ax4 = fig.add_subplot(gs[0, 3])
    plot_psf_profile(ref_data, ref_sources, ax4, 'Reference PSF Profile (Before)')
    
    # Row 2: Matched images
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(sci_matched, cmap='gray', origin='lower')
    ax5.set_title('Science Image (After Matching)')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(ref_matched, cmap='gray', origin='lower')
    ax6.set_title('Reference Image (After Matching)')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Row 2: PSF profiles after matching
    ax7 = fig.add_subplot(gs[1, 2])
    plot_psf_profile(sci_matched, sci_matched_sources, ax7, 'Science PSF Profile (After)')
    
    ax8 = fig.add_subplot(gs[1, 3])
    plot_psf_profile(ref_matched, ref_matched_sources, ax8, 'Reference PSF Profile (After)')
    
    # Row 3: Difference and statistics
    diff_before = sci_data - ref_data
    diff_after = sci_matched - ref_matched
    
    ax9 = fig.add_subplot(gs[2, 0])
    vmin, vmax = np.percentile(diff_before, [1, 99])
    im9 = ax9.imshow(diff_before, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    ax9.set_title('Difference (Before Matching)')
    plt.colorbar(im9, ax=ax9, fraction=0.046)
    
    ax10 = fig.add_subplot(gs[2, 1])
    vmin, vmax = np.percentile(diff_after, [1, 99])
    im10 = ax10.imshow(diff_after, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    ax10.set_title('Difference (After Matching)')
    plt.colorbar(im10, ax=ax10, fraction=0.046)
    
    # Histogram comparison
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.hist(diff_before.flatten(), bins=100, alpha=0.5, label='Before', color='blue')
    ax11.hist(diff_after.flatten(), bins=100, alpha=0.5, label='After', color='red')
    ax11.set_xlabel('Pixel Value')
    ax11.set_ylabel('Count')
    ax11.set_title('Difference Histogram')
    ax11.set_yscale('log')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # Statistics comparison
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    
    stats_text = "Statistics Comparison\n" + "="*30 + "\n\n"
    stats_text += "Before PSF Matching:\n"
    stats_text += f"  Diff Mean: {np.mean(diff_before):.4f}\n"
    stats_text += f"  Diff Std: {np.std(diff_before):.4f}\n"
    stats_text += f"  Diff RMS: {np.sqrt(np.mean(diff_before**2)):.4f}\n\n"
    
    stats_text += "After PSF Matching:\n"
    stats_text += f"  Diff Mean: {np.mean(diff_after):.4f}\n"
    stats_text += f"  Diff Std: {np.std(diff_after):.4f}\n"
    stats_text += f"  Diff RMS: {np.sqrt(np.mean(diff_after**2)):.4f}\n\n"
    
    improvement = (1 - np.std(diff_after)/np.std(diff_before)) * 100
    stats_text += f"Improvement: {improvement:.1f}%"
    
    ax12.text(0.1, 0.9, stats_text, transform=ax12.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    output_path = output_dir / 'psf_matching_comparison.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"\nSaved PSF comparison: {output_path}")
    plt.close()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize PSF matching results')
    parser.add_argument('--sci-file', type=str, help='Science image file')
    parser.add_argument('--ref-file', type=str, help='Reference image file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.sci_file and args.ref_file:
        sci_file = Path(args.sci_file)
        ref_file = Path(args.ref_file)
        output_dir = Path(args.output_dir) / sci_file.stem
    else:
        # Use default paths
        sci_dir = Path('../data_psf/sci_data')
        ref_dir = Path('../data_psf/dss_data')
        
        # Find first image pair
        sci_files = list(sci_dir.glob('*.jpg')) + list(sci_dir.glob('*.fits'))
        if not sci_files:
            print("No science images found!")
            return 1
        
        sci_file = sci_files[0]
        ref_file = ref_dir / sci_file.name
        
        if not ref_file.exists():
            print(f"Reference file not found: {ref_file}")
            return 1
        
        output_dir = Path('output') / sci_file.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_psf_comparison(sci_file, ref_file, output_dir)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

