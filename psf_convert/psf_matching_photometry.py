#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSF Matching and Photometry Extraction
Performs PSF matching, flux normalization, image subtraction, and photometry extraction
"""

import os
import sys
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve, Gaussian2DKernel
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from scipy import ndimage
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class PSFMatcher:
    """PSF matching and image subtraction class"""
    
    def __init__(self, sci_image_path, ref_image_path, output_dir='output'):
        """
        Initialize PSF matcher
        
        Parameters:
        -----------
        sci_image_path : str
            Path to science image
        ref_image_path : str
            Path to reference image (DSS)
        output_dir : str
            Output directory for results
        """
        self.sci_image_path = Path(sci_image_path)
        self.ref_image_path = Path(ref_image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sci_data = None
        self.ref_data = None
        self.diff_data = None
        
    def load_images(self):
        """Load science and reference images"""
        print(f"Loading science image: {self.sci_image_path}")
        print(f"Loading reference image: {self.ref_image_path}")
        
        # Load images (support both FITS and JPG)
        if self.sci_image_path.suffix.lower() in ['.fits', '.fit']:
            with fits.open(self.sci_image_path) as hdul:
                self.sci_data = hdul[0].data.astype(float)
        else:
            img = Image.open(self.sci_image_path).convert('L')
            self.sci_data = np.array(img, dtype=float)
        
        if self.ref_image_path.suffix.lower() in ['.fits', '.fit']:
            with fits.open(self.ref_image_path) as hdul:
                self.ref_data = hdul[0].data.astype(float)
        else:
            img = Image.open(self.ref_image_path).convert('L')
            self.ref_data = np.array(img, dtype=float)
        
        print(f"Science image shape: {self.sci_data.shape}")
        print(f"Reference image shape: {self.ref_data.shape}")
        
        # Ensure images have the same shape
        if self.sci_data.shape != self.ref_data.shape:
            print("Warning: Images have different shapes. Resizing reference to match science image.")
            from scipy.ndimage import zoom
            zoom_factors = (self.sci_data.shape[0] / self.ref_data.shape[0],
                          self.sci_data.shape[1] / self.ref_data.shape[1])
            self.ref_data = zoom(self.ref_data, zoom_factors, order=1)
        
        return self.sci_data, self.ref_data
    
    def estimate_psf_fwhm(self, data, nsigma=3.0):
        """
        Estimate PSF FWHM from image data
        
        Parameters:
        -----------
        data : ndarray
            Image data
        nsigma : float
            Detection threshold in sigma
            
        Returns:
        --------
        fwhm : float
            Estimated FWHM in pixels
        """
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
        # Detect sources
        daofind = DAOStarFinder(fwhm=3.0, threshold=nsigma*std)
        sources = daofind(data - median)
        
        if sources is None or len(sources) < 5:
            print("Warning: Not enough sources detected. Using default FWHM=3.0")
            return 3.0
        
        # Estimate FWHM from brightest sources
        sources.sort('flux')
        sources = sources[-20:]  # Use top 20 brightest sources
        
        fwhms = []
        for source in sources:
            x, y = int(source['xcentroid']), int(source['ycentroid'])
            if x < 10 or y < 10 or x > data.shape[1]-10 or y > data.shape[0]-10:
                continue
            
            # Extract small cutout
            cutout = data[y-10:y+10, x-10:x+10]
            if cutout.shape != (20, 20):
                continue
            
            # Estimate FWHM from cutout
            cutout = cutout - np.median(cutout)
            max_val = np.max(cutout)
            if max_val > 0:
                half_max = max_val / 2.0
                above_half = cutout > half_max
                fwhm_est = np.sqrt(np.sum(above_half) / np.pi) * 2.355
                if 1.0 < fwhm_est < 10.0:
                    fwhms.append(fwhm_est)
        
        if len(fwhms) > 0:
            fwhm = np.median(fwhms)
            print(f"Estimated FWHM: {fwhm:.2f} pixels")
            return fwhm
        else:
            print("Warning: Could not estimate FWHM. Using default FWHM=3.0")
            return 3.0
    
    def match_psf(self, target_fwhm=None):
        """
        Match PSF by convolving the sharper image

        Parameters:
        -----------
        target_fwhm : float, optional
            Target FWHM for matching. If None, uses the larger of the two FWHMs
        """
        print("\n=== PSF Matching ===")

        # Save original data for comparison
        self.sci_data_original = self.sci_data.copy()
        self.ref_data_original = self.ref_data.copy()

        # Estimate FWHM for both images
        sci_fwhm = self.estimate_psf_fwhm(self.sci_data)
        ref_fwhm = self.estimate_psf_fwhm(self.ref_data)

        # Store FWHM values
        self.sci_fwhm = sci_fwhm
        self.ref_fwhm = ref_fwhm

        print(f"Science image FWHM: {sci_fwhm:.2f} pixels")
        print(f"Reference image FWHM: {ref_fwhm:.2f} pixels")

        # Determine which image needs convolution
        # Use the smaller FWHM (sharper image) as target to preserve resolution
        if target_fwhm is None:
            target_fwhm = min(sci_fwhm, ref_fwhm)
            if sci_fwhm < ref_fwhm:
                print(f"Using science image PSF as target (sharper)")
            else:
                print(f"Using reference image PSF as target (sharper)")

        self.target_fwhm = target_fwhm
        print(f"Target FWHM: {target_fwhm:.2f} pixels")

        # Convolve the blurrier image to match the sharper one
        if sci_fwhm < ref_fwhm:
            # Science is sharper, convolve reference image
            conv_fwhm = np.sqrt(ref_fwhm**2 - sci_fwhm**2)
            sigma = conv_fwhm / 2.355
            kernel = Gaussian2DKernel(sigma)
            self.ref_data = convolve(self.ref_data, kernel, boundary='extend')
            print(f"Convolved reference image with sigma={sigma:.2f}")
        else:
            # Reference is sharper, convolve science image
            conv_fwhm = np.sqrt(sci_fwhm**2 - ref_fwhm**2)
            sigma = conv_fwhm / 2.355
            kernel = Gaussian2DKernel(sigma)
            self.sci_data = convolve(self.sci_data, kernel, boundary='extend')
            print(f"Convolved science image with sigma={sigma:.2f}")
    
    def normalize_flux(self, method='median'):
        """
        Normalize flux between science and reference images
        
        Parameters:
        -----------
        method : str
            Normalization method: 'median', 'mean', or 'mode'
        """
        print("\n=== Flux Normalization ===")
        
        # Calculate statistics
        sci_mean, sci_median, sci_std = sigma_clipped_stats(self.sci_data, sigma=3.0)
        ref_mean, ref_median, ref_std = sigma_clipped_stats(self.ref_data, sigma=3.0)
        
        print(f"Science image - mean: {sci_mean:.2f}, median: {sci_median:.2f}, std: {sci_std:.2f}")
        print(f"Reference image - mean: {ref_mean:.2f}, median: {ref_median:.2f}, std: {ref_std:.2f}")
        
        # Calculate scaling factor
        if method == 'median':
            scale_factor = sci_median / ref_median if ref_median != 0 else 1.0
        elif method == 'mean':
            scale_factor = sci_mean / ref_mean if ref_mean != 0 else 1.0
        else:
            scale_factor = 1.0
        
        print(f"Scaling reference image by factor: {scale_factor:.4f}")
        
        # Apply scaling
        self.ref_data = self.ref_data * scale_factor
        
        # Verify normalization
        ref_mean_new, ref_median_new, ref_std_new = sigma_clipped_stats(self.ref_data, sigma=3.0)
        print(f"After normalization - mean: {ref_mean_new:.2f}, median: {ref_median_new:.2f}, std: {ref_std_new:.2f}")
    
    def subtract_images(self):
        """Perform image subtraction"""
        print("\n=== Image Subtraction ===")
        
        self.diff_data = self.sci_data - self.ref_data
        
        # Calculate statistics
        diff_mean, diff_median, diff_std = sigma_clipped_stats(self.diff_data, sigma=3.0)
        print(f"Difference image - mean: {diff_mean:.2f}, median: {diff_median:.2f}, std: {diff_std:.2f}")
        
        return self.diff_data
    
    def extract_photometry(self, threshold=5.0, fwhm=3.0, output_catalog='photometry.txt'):
        """
        Extract photometry from difference image
        
        Parameters:
        -----------
        threshold : float
            Detection threshold in sigma
        fwhm : float
            FWHM for source detection
        output_catalog : str
            Output catalog filename
        """
        print("\n=== Photometry Extraction ===")
        
        # Calculate background statistics
        mean, median, std = sigma_clipped_stats(self.diff_data, sigma=3.0)
        print(f"Background - mean: {mean:.2f}, median: {median:.2f}, std: {std:.2f}")
        print(f"Detection threshold: {threshold} sigma = {threshold*std:.2f}")
        
        # Detect sources
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
        sources = daofind(self.diff_data - median)
        
        if sources is None or len(sources) == 0:
            print("No sources detected!")
            return None
        
        print(f"Detected {len(sources)} sources")
        
        # Perform aperture photometry
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        apertures = CircularAperture(positions, r=fwhm*2)
        phot_table = aperture_photometry(self.diff_data, apertures)
        
        # Combine results
        sources['aperture_flux'] = phot_table['aperture_sum']
        sources['mag'] = -2.5 * np.log10(np.abs(sources['aperture_flux']))
        
        # Save catalog
        output_path = self.output_dir / output_catalog
        sources.write(str(output_path), format='ascii.fixed_width', overwrite=True)
        print(f"Saved photometry catalog to: {output_path}")
        
        return sources
    
    def save_results(self):
        """Save all results"""
        print("\n=== Saving Results ===")

        # Save difference image as FITS
        diff_fits = self.output_dir / 'difference.fits'
        hdu = fits.PrimaryHDU(self.diff_data)
        hdu.writeto(str(diff_fits), overwrite=True)
        print(f"Saved difference image: {diff_fits}")

        # Save visualization
        self.plot_results()

        # Save PSF comparison if original data is available
        self.plot_psf_comparison()
    
    def plot_results(self):
        """Plot and save visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Science image
        im1 = axes[0, 0].imshow(self.sci_data, cmap='gray', origin='lower')
        axes[0, 0].set_title('Science Image')
        plt.colorbar(im1, ax=axes[0, 0])

        # Reference image
        im2 = axes[0, 1].imshow(self.ref_data, cmap='gray', origin='lower')
        axes[0, 1].set_title('Reference Image (Normalized)')
        plt.colorbar(im2, ax=axes[0, 1])

        # Difference image
        vmin, vmax = np.percentile(self.diff_data, [1, 99])
        im3 = axes[1, 0].imshow(self.diff_data, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('Difference Image')
        plt.colorbar(im3, ax=axes[1, 0])

        # Histogram
        axes[1, 1].hist(self.diff_data.flatten(), bins=100, alpha=0.7)
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Difference Image Histogram')
        axes[1, 1].set_yscale('log')

        plt.tight_layout()
        output_path = self.output_dir / 'results.png'
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {output_path}")
        plt.close()

    def plot_psf_comparison(self):
        """Plot PSF matching comparison with before/after"""
        if not hasattr(self, 'sci_data_original') or not hasattr(self, 'ref_data_original'):
            print("Warning: Original data not saved, skipping PSF comparison plot")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: Before PSF matching
        im1 = axes[0, 0].imshow(self.sci_data_original, cmap='gray', origin='lower')
        axes[0, 0].set_title(f'Science (Before)\nFWHM: {self.sci_fwhm:.2f}px')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

        im2 = axes[0, 1].imshow(self.ref_data_original, cmap='gray', origin='lower')
        axes[0, 1].set_title(f'Reference (Before)\nFWHM: {self.ref_fwhm:.2f}px')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

        diff_before = self.sci_data_original - self.ref_data_original
        vmin, vmax = np.percentile(diff_before, [1, 99])
        im3 = axes[0, 2].imshow(diff_before, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title(f'Difference (Before)\nStd: {np.std(diff_before):.2f}')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

        # Row 2: After PSF matching
        im4 = axes[1, 0].imshow(self.sci_data, cmap='gray', origin='lower')
        axes[1, 0].set_title(f'Science (After)\nTarget FWHM: {self.target_fwhm:.2f}px')
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

        im5 = axes[1, 1].imshow(self.ref_data, cmap='gray', origin='lower')
        axes[1, 1].set_title(f'Reference (After)\nTarget FWHM: {self.target_fwhm:.2f}px')
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

        diff_after = self.sci_data - self.ref_data
        vmin, vmax = np.percentile(diff_after, [1, 99])
        im6 = axes[1, 2].imshow(diff_after, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        improvement = (1 - np.std(diff_after)/np.std(diff_before)) * 100
        axes[1, 2].set_title(f'Difference (After)\nStd: {np.std(diff_after):.2f}\nImprovement: {improvement:.1f}%')
        plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

        plt.tight_layout()
        output_path = self.output_dir / 'psf_comparison.png'
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"Saved PSF comparison: {output_path}")
        plt.close()


def main():
    """Main function"""
    # Set paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data_psf'
    
    sci_dir = data_dir / 'sci_data'
    ref_dir = data_dir / 'dss_data'
    output_dir = script_dir / 'output'
    
    # Find matching files
    sci_files = list(sci_dir.glob('Npix*.jpg')) + list(sci_dir.glob('Npix*.fits'))
    ref_files = list(ref_dir.glob('Npix*.jpg')) + list(ref_dir.glob('Npix*.fits'))
    
    if not sci_files:
        print(f"Error: No science images found in {sci_dir}")
        return 1
    
    if not ref_files:
        print(f"Error: No reference images found in {ref_dir}")
        return 1
    
    print("="*80)
    print("PSF Matching and Photometry Extraction")
    print("="*80)
    print(f"Science images: {len(sci_files)}")
    print(f"Reference images: {len(ref_files)}")
    print("="*80)
    
    # Process each pair
    for sci_file in sci_files:
        # Find matching reference file
        ref_file = ref_dir / sci_file.name
        if not ref_file.exists():
            # Try different extensions
            ref_file = ref_dir / (sci_file.stem + '.jpg')
            if not ref_file.exists():
                ref_file = ref_dir / (sci_file.stem + '.fits')
        
        if not ref_file.exists():
            print(f"Warning: No matching reference file for {sci_file.name}")
            continue
        
        print(f"\nProcessing: {sci_file.name}")
        
        # Create output directory for this pair
        pair_output = output_dir / sci_file.stem
        
        # Initialize PSF matcher
        matcher = PSFMatcher(sci_file, ref_file, pair_output)
        
        # Load images
        matcher.load_images()
        
        # PSF matching
        matcher.match_psf()
        
        # Flux normalization
        matcher.normalize_flux(method='median')
        
        # Image subtraction
        matcher.subtract_images()
        
        # Extract photometry
        sources = matcher.extract_photometry(threshold=5.0, fwhm=3.0)
        
        # Save results
        matcher.save_results()
        
        print(f"Completed processing: {sci_file.name}")
    
    print("\n" + "="*80)
    print("All processing completed!")
    print(f"Results saved to: {output_dir}")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

