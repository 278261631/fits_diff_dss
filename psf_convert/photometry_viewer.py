#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photometry Viewer with Interactive Matplotlib UI
Displays DSS reference image and photometry light curves for clicked sources
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import warnings
warnings.filterwarnings('ignore')


class PhotometryData:
    """Class to store and manage photometry data"""
    
    def __init__(self):
        self.datasets = {}  # {dataset_name: {tile_name: data}}
        self.ref_image = None
        self.ref_image_path = None
        
    def load_photometry_file(self, filepath):
        """Load a single photometry file"""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Try pipe-separated format first
                if '|' in line:
                    parts = [p.strip() for p in line.split('|') if p.strip()]

                    # Skip header line (contains text like "xcentroid")
                    if len(parts) > 0 and not parts[0].replace('.', '').replace('-', '').isdigit():
                        continue

                    if len(parts) >= 11:
                        try:
                            # parts[0]=id, parts[1]=xcentroid, parts[2]=ycentroid,
                            # parts[9]=flux, parts[10]=mag
                            x = float(parts[1])  # xcentroid
                            y = float(parts[2])  # ycentroid
                            flux = float(parts[9])  # flux
                            mag = float(parts[10])  # mag
                            data.append({'x': x, 'y': y, 'flux': flux, 'mag': mag})
                        except (ValueError, IndexError):
                            continue
                else:
                    # Try space-separated format
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            flux = float(parts[2])
                            mag = float(parts[3])
                            data.append({'x': x, 'y': y, 'flux': flux, 'mag': mag})
                        except ValueError:
                            continue
        return data
    
    def load_all_data(self, output_dir, ref_image_path):
        """Load all photometry data from output directory"""
        output_path = Path(output_dir)
        self.ref_image_path = Path(ref_image_path)
        
        # Load reference image
        if self.ref_image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            self.ref_image = np.array(Image.open(self.ref_image_path).convert('L'))
        else:
            from astropy.io import fits
            with fits.open(self.ref_image_path) as hdul:
                self.ref_image = hdul[0].data
        
        # Find all sci_data directories
        for sci_dir in output_path.iterdir():
            if not sci_dir.is_dir() or not sci_dir.name.startswith('sci_data'):
                continue
            
            dataset_name = sci_dir.name
            self.datasets[dataset_name] = {}
            
            # Find all tile directories
            for tile_dir in sci_dir.iterdir():
                if not tile_dir.is_dir():
                    continue
                
                tile_name = tile_dir.name
                phot_file = tile_dir / 'photometry.txt'
                
                if phot_file.exists():
                    data = self.load_photometry_file(phot_file)
                    if data:
                        self.datasets[dataset_name][tile_name] = data
        
        return len(self.datasets) > 0
    
    def find_nearest_source(self, x, y, dataset_name, tile_name, max_distance=10):
        """Find the nearest source to clicked position"""
        if dataset_name not in self.datasets:
            return None
        if tile_name not in self.datasets[dataset_name]:
            return None
        
        data = self.datasets[dataset_name][tile_name]
        min_dist = float('inf')
        nearest = None
        
        for source in data:
            dist = np.sqrt((source['x'] - x)**2 + (source['y'] - y)**2)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                nearest = source
        
        return nearest
    
    def get_light_curve(self, x, y, tile_name, max_distance=10):
        """Get light curve for a source across all datasets"""
        light_curve = []
        
        for dataset_name in sorted(self.datasets.keys()):
            if tile_name not in self.datasets[dataset_name]:
                continue
            
            source = self.find_nearest_source(x, y, dataset_name, tile_name, max_distance)
            if source:
                light_curve.append({
                    'dataset': dataset_name,
                    'x': source['x'],
                    'y': source['y'],
                    'flux': source['flux'],
                    'mag': source['mag']
                })
        
        return light_curve


class PhotometryViewer:
    """Interactive photometry viewer using matplotlib"""

    def __init__(self, output_dir, ref_image_path):
        """
        Initialize viewer

        Parameters:
        -----------
        output_dir : str or Path
            Output directory containing photometry data
        ref_image_path : str or Path
            Path to DSS reference image
        """
        self.phot_data = PhotometryData()
        self.current_tile = None
        self.selected_source = None

        # Load data
        print("Loading data...")
        success = self.phot_data.load_all_data(output_dir, ref_image_path)

        if not success:
            print("Error: No photometry data found!")
            return

        # Get available tiles
        self.tiles = set()
        for dataset in self.phot_data.datasets.values():
            self.tiles.update(dataset.keys())
        self.tiles = sorted(self.tiles)

        if not self.tiles:
            print("Error: No tiles found!")
            return

        self.current_tile = self.tiles[0]

        # Print summary
        n_datasets = len(self.phot_data.datasets)
        n_tiles = len(self.tiles)
        print(f"Loaded {n_datasets} datasets, {n_tiles} tile(s)")
        print(f"Tiles: {', '.join(self.tiles)}")
        print()

        # Create figure
        self.fig = plt.figure(figsize=(14, 6))
        self.fig.canvas.manager.set_window_title('Photometry Viewer')

        # Connect click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Create UI
        self.create_ui()

    def create_ui(self):
        """Create user interface"""
        # Create subplots
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                                   hspace=0.3, wspace=0.3)

        # Image axis (left, spans both rows)
        self.ax_image = self.fig.add_subplot(gs[:, 0])

        # Light curve axes (right)
        self.ax_flux = self.fig.add_subplot(gs[0, 1])
        self.ax_mag = self.fig.add_subplot(gs[1, 1])

        # Display initial image
        self.display_reference_image()

        # Initial light curve placeholder
        self.clear_light_curve()

        # Add title
        self.fig.suptitle('Photometry Viewer - Click on a source to view its light curve',
                         fontsize=12, fontweight='bold')


    def display_reference_image(self):
        """Display reference image with sources"""
        if self.phot_data.ref_image is None:
            return

        self.ax_image.clear()

        # Display image
        self.ax_image.imshow(self.phot_data.ref_image, cmap='gray', origin='lower')

        # Overlay sources from first dataset
        if self.current_tile and self.phot_data.datasets:
            first_dataset = sorted(self.phot_data.datasets.keys())[0]
            if self.current_tile in self.phot_data.datasets[first_dataset]:
                sources = self.phot_data.datasets[first_dataset][self.current_tile]
                x_coords = [s['x'] for s in sources]
                y_coords = [s['y'] for s in sources]
                self.ax_image.scatter(x_coords, y_coords, c='red', s=50, alpha=0.5,
                                     marker='o', facecolors='none', edgecolors='red', linewidths=1)

        self.ax_image.set_xlabel('X (pixels)', fontsize=10)
        self.ax_image.set_ylabel('Y (pixels)', fontsize=10)
        self.ax_image.set_title(f'DSS Reference Image - {self.current_tile}', fontsize=11)

        self.fig.canvas.draw()

    def on_click(self, event):
        """Handle mouse click on image"""
        if event.inaxes != self.ax_image or self.current_tile is None:
            return

        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata

        # Find nearest source
        first_dataset = sorted(self.phot_data.datasets.keys())[0]
        source = self.phot_data.find_nearest_source(x, y, first_dataset, self.current_tile)

        if source is None:
            print(f'No source found near ({x:.1f}, {y:.1f})')
            return

        self.selected_source = source
        print(f'Selected source at ({source["x"]:.1f}, {source["y"]:.1f}), mag={source["mag"]:.2f}')

        # Get and display light curve
        light_curve = self.phot_data.get_light_curve(
            source['x'], source['y'], self.current_tile
        )

        if light_curve:
            self.display_light_curve(light_curve)
        else:
            print('No light curve data found for this source')

    def display_light_curve(self, light_curve):
        """Display light curve"""
        self.ax_flux.clear()
        self.ax_mag.clear()

        datasets = [lc['dataset'] for lc in light_curve]
        fluxes = [lc['flux'] for lc in light_curve]
        mags = [lc['mag'] for lc in light_curve]

        # Plot flux
        self.ax_flux.plot(range(len(datasets)), fluxes, 'o-', color='blue', markersize=8, linewidth=2)
        self.ax_flux.set_ylabel('Flux', fontsize=11)
        self.ax_flux.set_title(f'Source at ({light_curve[0]["x"]:.1f}, {light_curve[0]["y"]:.1f})',
                              fontsize=11)
        self.ax_flux.grid(True, alpha=0.3)
        self.ax_flux.set_xticks(range(len(datasets)))
        self.ax_flux.set_xticklabels([])

        # Plot magnitude
        self.ax_mag.plot(range(len(datasets)), mags, 'o-', color='red', markersize=8, linewidth=2)
        self.ax_mag.set_xlabel('Dataset', fontsize=11)
        self.ax_mag.set_ylabel('Magnitude', fontsize=11)
        self.ax_mag.invert_yaxis()
        self.ax_mag.grid(True, alpha=0.3)
        self.ax_mag.set_xticks(range(len(datasets)))
        self.ax_mag.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)

        self.fig.canvas.draw()

    def clear_light_curve(self):
        """Clear light curve plot"""
        self.ax_flux.clear()
        self.ax_mag.clear()

        self.ax_flux.text(0.5, 0.5, 'Click on a source\nto view its light curve',
                         ha='center', va='center', fontsize=12, color='gray',
                         transform=self.ax_flux.transAxes)
        self.ax_flux.set_xlim(0, 1)
        self.ax_flux.set_ylim(0, 1)
        self.ax_flux.axis('off')

        self.ax_mag.axis('off')

        self.fig.canvas.draw()

    def show(self):
        """Show the viewer"""
        plt.show()


def main():
    """Main function"""
    # Set paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'output'
    ref_image = script_dir.parent / 'data_psf' / 'dss_data' / 'Npix512845.jpg'

    # Check if paths exist
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        print("Please run psf_matching_photometry.py first to generate data.")
        return 1

    if not ref_image.exists():
        print(f"Error: Reference image not found: {ref_image}")
        print("Please check the data_psf/dss_data/ directory.")
        return 1

    print("="*80)
    print("Photometry Viewer")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Reference image: {ref_image}")
    print()

    # Create and show viewer
    viewer = PhotometryViewer(output_dir, ref_image)

    print()
    print("Instructions:")
    print("  - Click on any red circle (source) to view its light curve")
    print("  - The light curve shows flux and magnitude across all datasets")
    print("  - Close the window to exit")
    print()

    viewer.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())

