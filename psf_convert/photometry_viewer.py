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

# Set matplotlib backend to display in separate window (not PyCharm Plots panel)
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive window

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import warnings
warnings.filterwarnings('ignore')

# Import analysis libraries
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Try to import DTW library
try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("Warning: dtaidistance not available. DTW analysis will be disabled.")


class TrendAnalyzer:
    """Class for analyzing light curve trends"""

    def __init__(self):
        self.methods = {
            'None': 'No classification',
            'Z-Score': 'Z-score outlier detection',
            'Clustering': 'Hierarchical clustering',
            'PCA': 'Principal Component Analysis',
            'Change Point': 'Change point detection',
            'DTW': 'Dynamic Time Warping clustering'
        }

    def analyze(self, light_curves, method='None', n_clusters=3):
        """
        Analyze light curves and return classification labels

        Parameters:
        -----------
        light_curves : list of arrays
            List of light curves (each is array of flux values)
        method : str
            Analysis method to use
        n_clusters : int
            Number of clusters for clustering methods

        Returns:
        --------
        labels : array
            Classification label for each light curve
        colors : list
            Color for each class
        """
        if method == 'None' or len(light_curves) == 0:
            return np.zeros(len(light_curves), dtype=int), ['blue']

        # Convert to numpy array
        data = np.array(light_curves)
        n_sources = len(data)

        if method == 'Z-Score':
            return self._zscore_analysis(data)
        elif method == 'Clustering':
            return self._hierarchical_clustering(data, n_clusters)
        elif method == 'PCA':
            return self._pca_analysis(data, n_clusters)
        elif method == 'Change Point':
            return self._change_point_analysis(data)
        elif method == 'DTW':
            return self._dtw_clustering(data, n_clusters)
        else:
            return np.zeros(n_sources, dtype=int), ['blue']

    def _zscore_analysis(self, data):
        """Z-score based outlier detection"""
        # Calculate variance for each light curve
        variances = np.var(data, axis=1)

        # Calculate z-scores
        z_scores = np.abs(stats.zscore(variances))

        # Classify: 0=normal, 1=moderate variation, 2=high variation
        labels = np.zeros(len(data), dtype=int)
        labels[z_scores > 1] = 1
        labels[z_scores > 2] = 2

        colors = ['blue', 'orange', 'red']
        return labels, colors

    def _hierarchical_clustering(self, data, n_clusters):
        """Hierarchical clustering"""
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Perform hierarchical clustering
        linkage_matrix = linkage(data_scaled, method='ward')
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1

        # Generate colors
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        return labels, colors

    def _pca_analysis(self, data, n_clusters):
        """PCA-based clustering"""
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Apply PCA
        pca = PCA(n_components=min(2, data.shape[1]))
        data_pca = pca.fit_transform(data_scaled)

        # K-means clustering on PCA components
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_pca)

        # Generate colors
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        return labels, colors

    def _change_point_analysis(self, data):
        """Change point detection"""
        labels = np.zeros(len(data), dtype=int)

        for i, curve in enumerate(data):
            # Calculate differences between consecutive points
            diffs = np.abs(np.diff(curve))

            # Detect significant changes
            threshold = np.mean(diffs) + 2 * np.std(diffs)
            n_changes = np.sum(diffs > threshold)

            # Classify: 0=stable, 1=moderate changes, 2=many changes
            if n_changes == 0:
                labels[i] = 0
            elif n_changes <= 2:
                labels[i] = 1
            else:
                labels[i] = 2

        colors = ['blue', 'orange', 'red']
        return labels, colors

    def _dtw_clustering(self, data, n_clusters):
        """DTW-based clustering"""
        if not DTW_AVAILABLE:
            print("DTW library not available, using hierarchical clustering instead")
            return self._hierarchical_clustering(data, n_clusters)

        n_sources = len(data)

        # Calculate DTW distance matrix
        dist_matrix = np.zeros((n_sources, n_sources))
        for i in range(n_sources):
            for j in range(i+1, n_sources):
                distance = dtw.distance(data[i], data[j])
                dist_matrix[i, j] = distance
                dist_matrix[j, i] = distance

        # Perform hierarchical clustering on distance matrix
        linkage_matrix = linkage(dist_matrix, method='average')
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1

        # Generate colors
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        return labels, colors


class PhotometryData:
    """Class to store and manage photometry data"""
    
    def __init__(self):
        self.datasets = {}  # {dataset_name: {tile_name: data}}
        self.ref_images = {}  # {tile_name: image_array}
        self.sci_images = {}  # {dataset_name: {tile_name: image_array}}
        
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
    
    def load_all_data(self, output_dir, ref_image_dir, sci_image_dir):
        """Load all photometry data from output directory"""
        output_path = Path(output_dir)
        self.ref_image_dir = Path(ref_image_dir)
        self.sci_image_dir = Path(sci_image_dir)

        # Find all sci_data directories
        for sci_dir in output_path.iterdir():
            if not sci_dir.is_dir() or not sci_dir.name.startswith('sci_data'):
                continue

            dataset_name = sci_dir.name
            self.datasets[dataset_name] = {}
            self.sci_images[dataset_name] = {}

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

                        # Load reference image for this tile if not already loaded
                        if tile_name not in self.ref_images:
                            ref_img_path = self.ref_image_dir / f'{tile_name}.jpg'
                            if ref_img_path.exists():
                                self.ref_images[tile_name] = np.array(Image.open(ref_img_path).convert('L'))
                            else:
                                # Try other extensions
                                for ext in ['.png', '.jpeg', '.fits']:
                                    ref_img_path = self.ref_image_dir / f'{tile_name}{ext}'
                                    if ref_img_path.exists():
                                        if ext == '.fits':
                                            from astropy.io import fits
                                            with fits.open(ref_img_path) as hdul:
                                                self.ref_images[tile_name] = hdul[0].data
                                        else:
                                            self.ref_images[tile_name] = np.array(Image.open(ref_img_path).convert('L'))
                                        break

                        # Load science image for this dataset and tile
                        sci_img_path = self.sci_image_dir / dataset_name / f'{tile_name}.jpg'
                        if sci_img_path.exists():
                            self.sci_images[dataset_name][tile_name] = np.array(Image.open(sci_img_path).convert('L'))
                        else:
                            # Try other extensions
                            for ext in ['.png', '.jpeg', '.fits']:
                                sci_img_path = self.sci_image_dir / dataset_name / f'{tile_name}{ext}'
                                if sci_img_path.exists():
                                    if ext == '.fits':
                                        from astropy.io import fits
                                        with fits.open(sci_img_path) as hdul:
                                            self.sci_images[dataset_name][tile_name] = hdul[0].data
                                    else:
                                        self.sci_images[dataset_name][tile_name] = np.array(Image.open(sci_img_path).convert('L'))
                                    break

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

    def __init__(self, output_dir, ref_image_dir, sci_image_dir):
        """
        Initialize viewer

        Parameters:
        -----------
        output_dir : str or Path
            Output directory containing photometry data
        ref_image_dir : str or Path
            Directory containing DSS reference images
        sci_image_dir : str or Path
            Directory containing science images
        """
        self.phot_data = PhotometryData()
        self.trend_analyzer = TrendAnalyzer()
        self.current_tile = None
        self.current_image_type = 'dss'  # 'dss' or dataset name
        self.current_analysis_method = 'None'
        self.n_clusters = 3
        self.selected_source = None

        # Load data
        print("Loading data...")
        success = self.phot_data.load_all_data(output_dir, ref_image_dir, sci_image_dir)

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

        # Get available datasets for image switching
        self.datasets = sorted(self.phot_data.datasets.keys())

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
                                   hspace=0.3, wspace=0.3, top=0.92, bottom=0.08, left=0.15)

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

        # Add tile selector (radio buttons) if multiple tiles
        if len(self.tiles) > 1:
            ax_radio_tile = plt.axes([0.01, 0.5, 0.12, 0.15])
            self.radio_tile = RadioButtons(ax_radio_tile, self.tiles, active=0)
            self.radio_tile.on_clicked(self.on_tile_changed)
            # Add label
            ax_radio_tile.text(0.5, 1.1, 'Select Tile:', transform=ax_radio_tile.transAxes,
                         ha='center', fontsize=10, fontweight='bold')

        # Add image selector (radio buttons)
        image_options = ['DSS (Reference)'] + self.datasets
        ax_radio_img = plt.axes([0.01, 0.25, 0.12, 0.20])
        self.radio_img = RadioButtons(ax_radio_img, image_options, active=0)
        self.radio_img.on_clicked(self.on_image_changed)
        # Add label
        ax_radio_img.text(0.5, 1.05, 'Select Image:', transform=ax_radio_img.transAxes,
                         ha='center', fontsize=9, fontweight='bold')

        # Add analysis method selector (radio buttons)
        analysis_options = list(self.trend_analyzer.methods.keys())
        ax_radio_analysis = plt.axes([0.01, 0.08, 0.12, 0.15])
        self.radio_analysis = RadioButtons(ax_radio_analysis, analysis_options, active=0)
        self.radio_analysis.on_clicked(self.on_analysis_changed)
        # Add label
        ax_radio_analysis.text(0.5, 1.1, 'Trend Analysis:', transform=ax_radio_analysis.transAxes,
                         ha='center', fontsize=9, fontweight='bold')

        # Add button for showing all sources
        ax_button = plt.axes([0.35, 0.01, 0.15, 0.04])
        self.btn_all = Button(ax_button, 'Show All Sources')
        self.btn_all.on_clicked(self.show_all_sources)

        # Add buttons for adjusting cluster number
        ax_button_minus = plt.axes([0.52, 0.01, 0.05, 0.04])
        self.btn_minus = Button(ax_button_minus, '-')
        self.btn_minus.on_clicked(self.decrease_clusters)

        ax_button_plus = plt.axes([0.58, 0.01, 0.05, 0.04])
        self.btn_plus = Button(ax_button_plus, '+')
        self.btn_plus.on_clicked(self.increase_clusters)

        # Add text to show current cluster number
        self.cluster_text = self.fig.text(0.65, 0.03, f'Clusters: {self.n_clusters}',
                                          fontsize=10, ha='left')


    def display_reference_image(self):
        """Display current image (DSS or science) with sources"""
        self.ax_image.clear()

        # Determine which image to display
        if self.current_image_type == 'dss':
            if self.current_tile not in self.phot_data.ref_images:
                return
            image = self.phot_data.ref_images[self.current_tile]
            title = f'DSS Reference Image - {self.current_tile}'
        else:
            # Display science image
            dataset_name = self.current_image_type
            if dataset_name not in self.phot_data.sci_images:
                return
            if self.current_tile not in self.phot_data.sci_images[dataset_name]:
                return
            image = self.phot_data.sci_images[dataset_name][self.current_tile]
            title = f'{dataset_name} - {self.current_tile}'

        # Display image
        self.ax_image.imshow(image, cmap='gray', origin='lower')

        # Overlay sources from first dataset (or current dataset if viewing science image)
        if self.current_tile and self.phot_data.datasets:
            if self.current_image_type == 'dss':
                source_dataset = sorted(self.phot_data.datasets.keys())[0]
            else:
                source_dataset = self.current_image_type

            if source_dataset in self.phot_data.datasets and self.current_tile in self.phot_data.datasets[source_dataset]:
                sources = self.phot_data.datasets[source_dataset][self.current_tile]
                x_coords = [s['x'] for s in sources]
                y_coords = [s['y'] for s in sources]
                self.ax_image.scatter(x_coords, y_coords, c='red', s=50, alpha=0.5,
                                     marker='o', facecolors='none', edgecolors='red', linewidths=1)

        self.ax_image.set_xlabel('X (pixels)', fontsize=10)
        self.ax_image.set_ylabel('Y (pixels)', fontsize=10)
        self.ax_image.set_title(title, fontsize=11)

        self.fig.canvas.draw()

    def on_tile_changed(self, label):
        """Handle tile selection change"""
        self.current_tile = label
        print(f"\nSwitched to tile: {self.current_tile}")

        # Update display
        self.display_reference_image()
        self.clear_light_curve()
        self.selected_source = None

    def on_image_changed(self, label):
        """Handle image selection change"""
        if label == 'DSS (Reference)':
            self.current_image_type = 'dss'
            print(f"\nSwitched to DSS reference image")
        else:
            self.current_image_type = label
            print(f"\nSwitched to {label} image")

        # Update display
        self.display_reference_image()

    def on_analysis_changed(self, label):
        """Handle analysis method change"""
        self.current_analysis_method = label
        print(f"\nAnalysis method changed to: {label}")
        print(f"  {self.trend_analyzer.methods[label]}")

    def decrease_clusters(self, event):
        """Decrease number of clusters"""
        if self.n_clusters > 2:
            self.n_clusters -= 1
            self.cluster_text.set_text(f'Clusters: {self.n_clusters}')
            self.fig.canvas.draw()
            print(f"\nCluster number decreased to: {self.n_clusters}")

    def increase_clusters(self, event):
        """Increase number of clusters"""
        if self.n_clusters < 10:
            self.n_clusters += 1
            self.cluster_text.set_text(f'Clusters: {self.n_clusters}')
            self.fig.canvas.draw()
            print(f"\nCluster number increased to: {self.n_clusters}")

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

    def show_all_sources(self, event=None):
        """Show all sources' light curves in a new window with classification"""
        if not self.current_tile or not self.phot_data.datasets:
            print("No data to display")
            return

        print("\nGenerating all sources plot...")
        print(f"Analysis method: {self.current_analysis_method}")

        # Get all sources from first dataset
        first_dataset = sorted(self.phot_data.datasets.keys())[0]
        if self.current_tile not in self.phot_data.datasets[first_dataset]:
            print("No sources found in current tile")
            return

        sources = self.phot_data.datasets[first_dataset][self.current_tile]
        n_sources = len(sources)

        print(f"Processing {n_sources} sources...")

        # Get dataset names
        dataset_names = sorted(self.phot_data.datasets.keys())
        n_datasets = len(dataset_names)
        x_indices = range(n_datasets)

        # Collect all light curves
        all_light_curves = []
        valid_sources = []
        incomplete_indices = []  # Track which sources have incomplete data

        for idx, source in enumerate(sources):
            light_curve = self.phot_data.get_light_curve(
                source['x'], source['y'], self.current_tile
            )

            if light_curve and len(light_curve) > 0:
                # Create flux array with 0 for missing data
                fluxes = []
                dataset_dict = {lc['dataset']: lc['flux'] for lc in light_curve}

                for dataset_name in dataset_names:
                    if dataset_name in dataset_dict:
                        fluxes.append(dataset_dict[dataset_name])
                    else:
                        fluxes.append(0)  # Fill missing data with 0

                all_light_curves.append(fluxes)
                valid_sources.append(source)

                # Track if this source has incomplete data
                if len(light_curve) < n_datasets:
                    incomplete_indices.append(len(valid_sources) - 1)

        n_plotted = len(all_light_curves)
        n_skipped = n_sources - n_plotted
        n_incomplete = len(incomplete_indices)

        if n_plotted == 0:
            print("No sources found")
            return

        # Print data summary
        print(f"Data summary:")
        print(f"  Total sources: {n_plotted}")
        print(f"  Complete data (4 points): {n_plotted - n_incomplete}")
        print(f"  Incomplete data (filled with 0): {n_incomplete}")
        if n_incomplete > 0:
            print(f"  Incomplete source indices: {incomplete_indices[:10]}{'...' if len(incomplete_indices) > 10 else ''}")

        # Perform trend analysis
        print(f"\nPerforming {self.current_analysis_method} analysis with {self.n_clusters} clusters...")
        labels, colors = self.trend_analyzer.analyze(
            all_light_curves,
            method=self.current_analysis_method,
            n_clusters=self.n_clusters
        )

        # Count sources in each class and check incomplete distribution
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        print(f"Classification complete: {n_classes} classes found")
        for label in unique_labels:
            count = np.sum(labels == label)
            # Count how many incomplete sources in this class
            incomplete_in_class = sum(1 for idx in incomplete_indices if labels[idx] == label)
            print(f"  Class {label}: {count} sources ({incomplete_in_class} incomplete)")

        # Create new figure
        fig_all = plt.figure(figsize=(16, 9))
        fig_all.canvas.manager.set_window_title('All Sources Light Curves - Classified')

        # Create subplots with legend space
        ax_flux_all = fig_all.add_subplot(211)
        ax_mag_all = fig_all.add_subplot(212)

        # Plot each source with its class color
        for i, (fluxes, source) in enumerate(zip(all_light_curves, valid_sources)):
            label_idx = labels[i]
            color = colors[label_idx] if isinstance(colors[label_idx], str) else colors[label_idx]

            # Get magnitudes (with 0 for missing data)
            light_curve = self.phot_data.get_light_curve(
                source['x'], source['y'], self.current_tile
            )
            dataset_dict = {lc['dataset']: lc['mag'] for lc in light_curve}
            mags = []
            for dataset_name in dataset_names:
                if dataset_name in dataset_dict:
                    mags.append(dataset_dict[dataset_name])
                else:
                    mags.append(0)  # Fill missing magnitude with 0

            # Use different style for incomplete data
            if i in incomplete_indices:
                # Incomplete data: dashed line, lower alpha
                ax_flux_all.plot(x_indices, fluxes, 'o--', color=color,
                               alpha=0.3, linewidth=0.6, markersize=2)
                ax_mag_all.plot(x_indices, mags, 'o--', color=color,
                              alpha=0.3, linewidth=0.6, markersize=2)
            else:
                # Complete data: solid line
                ax_flux_all.plot(x_indices, fluxes, 'o-', color=color,
                               alpha=0.4, linewidth=0.8, markersize=3)
                ax_mag_all.plot(x_indices, mags, 'o-', color=color,
                              alpha=0.4, linewidth=0.8, markersize=3)

        # Add legend for classes
        from matplotlib.patches import Patch
        legend_elements = []
        for label in unique_labels:
            color = colors[label] if isinstance(colors[label], str) else colors[label]
            count = np.sum(labels == label)
            legend_elements.append(Patch(facecolor=color, label=f'Class {label} ({count})'))

        ax_flux_all.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # Format flux plot
        ax_flux_all.set_ylabel('Flux', fontsize=12)
        title = f'All Sources Light Curves - {self.current_tile}\n'
        title += f'Method: {self.current_analysis_method} | '
        title += f'{n_plotted} sources, {n_classes} classes'
        if n_incomplete > 0:
            title += f' | {n_incomplete} incomplete (dashed)'
        if n_skipped > 0:
            title += f' | {n_skipped} skipped'
        ax_flux_all.set_title(title, fontsize=13, fontweight='bold')
        ax_flux_all.grid(True, alpha=0.3)
        ax_flux_all.set_xticks(x_indices)
        ax_flux_all.set_xticklabels([])

        # Format magnitude plot
        ax_mag_all.set_xlabel('Dataset', fontsize=12)
        ax_mag_all.set_ylabel('Magnitude', fontsize=12)
        ax_mag_all.invert_yaxis()
        ax_mag_all.grid(True, alpha=0.3)
        ax_mag_all.set_xticks(x_indices)
        ax_mag_all.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=10)

        fig_all.tight_layout()

        print(f"✓ Plotted {n_plotted} sources")
        if n_incomplete > 0:
            print(f"  {n_incomplete} sources with incomplete data (shown as dashed lines, missing values filled with 0)")
        if n_skipped > 0:
            print(f"  {n_skipped} sources skipped (no data)")
        print("Close the window to return to main viewer")

        plt.show()

    def show(self):
        """Show the viewer"""
        plt.show()


def main():
    """Main function"""
    # Set paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'output'
    ref_image_dir = script_dir.parent / 'data_psf' / 'dss_data'
    sci_image_dir = script_dir.parent / 'data_psf'

    # Check if paths exist
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        print("Please run psf_matching_photometry.py first to generate data.")
        return 1

    if not ref_image_dir.exists():
        print(f"Error: Reference image directory not found: {ref_image_dir}")
        print("Please check the data_psf/dss_data/ directory.")
        return 1

    if not sci_image_dir.exists():
        print(f"Error: Science image directory not found: {sci_image_dir}")
        print("Please check the data_psf/ directory.")
        return 1

    print("="*80)
    print("Photometry Viewer")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Reference image directory: {ref_image_dir}")
    print(f"Science image directory: {sci_image_dir}")
    print()

    # Create and show viewer
    viewer = PhotometryViewer(output_dir, ref_image_dir, sci_image_dir)

    print()
    print("Instructions:")
    print("  - Use 'Select Tile' radio buttons to switch between tiles (if multiple)")
    print("  - Use 'Select Image' radio buttons to switch between DSS and science images")
    print("  - Click on any red circle (source) to view its light curve")
    print("  - Click 'Show All Sources' button to see all sources in current tile")
    print("  - The light curve shows flux and magnitude across all datasets")
    print("  - Close the window to exit")
    print()

    viewer.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())

