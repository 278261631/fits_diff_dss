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
import pickle
import json

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
            'DTW': 'Dynamic Time Warping clustering',
            'DTW-Shape': 'DTW with shape normalization',
            'Correlation': 'Correlation-based shape clustering',
            'Derivative': 'Derivative-based shape clustering'
        }

    def estimate_optimal_clusters(self, light_curves, method='silhouette', max_clusters=10):
        """
        Estimate optimal number of clusters

        Parameters:
        -----------
        light_curves : list of arrays
            List of light curves
        method : str
            Method to use: 'silhouette', 'elbow', or 'gap'
        max_clusters : int
            Maximum number of clusters to test

        Returns:
        --------
        optimal_k : int
            Estimated optimal number of clusters
        """
        if len(light_curves) < 4:
            return 2

        data = np.array(light_curves)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        max_k = min(max_clusters, len(light_curves) - 1)

        if method == 'silhouette':
            return self._silhouette_method(data_scaled, max_k)
        elif method == 'elbow':
            return self._elbow_method(data_scaled, max_k)
        elif method == 'gap':
            return self._gap_statistic(data_scaled, max_k)
        else:
            return 3  # Default

    def _silhouette_method(self, data, max_k):
        """Use silhouette score to find optimal k"""
        from sklearn.metrics import silhouette_score

        silhouette_scores = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)

        # Find k with highest silhouette score
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = k_range[optimal_idx]

        print(f"  Silhouette scores: {[f'{s:.3f}' for s in silhouette_scores]}")
        print(f"  Optimal k by silhouette: {optimal_k} (score: {silhouette_scores[optimal_idx]:.3f})")

        return optimal_k

    def _elbow_method(self, data, max_k):
        """Use elbow method (inertia) to find optimal k"""
        inertias = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)

        # Find elbow point using second derivative
        if len(inertias) < 3:
            return 2

        # Calculate rate of change
        deltas = np.diff(inertias)
        second_deltas = np.diff(deltas)

        # Find point where second derivative is maximum (sharpest bend)
        elbow_idx = np.argmax(np.abs(second_deltas)) + 2  # +2 because of two diff operations
        optimal_k = k_range[min(elbow_idx, len(k_range) - 1)]

        print(f"  Inertias: {[f'{i:.1f}' for i in inertias]}")
        print(f"  Optimal k by elbow: {optimal_k}")

        return optimal_k

    def _gap_statistic(self, data, max_k):
        """Use gap statistic to find optimal k"""
        n_refs = 10  # Number of reference datasets
        gaps = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            # Cluster original data
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            orig_disp = kmeans.inertia_

            # Generate reference datasets and cluster them
            ref_disps = []
            for _ in range(n_refs):
                # Generate random data with same bounds
                random_data = np.random.uniform(
                    low=data.min(axis=0),
                    high=data.max(axis=0),
                    size=data.shape
                )
                kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_ref.fit(random_data)
                ref_disps.append(kmeans_ref.inertia_)

            # Calculate gap
            gap = np.log(np.mean(ref_disps)) - np.log(orig_disp)
            gaps.append(gap)

        # Find k where gap is maximum
        optimal_idx = np.argmax(gaps)
        optimal_k = k_range[optimal_idx]

        print(f"  Gap statistics: {[f'{g:.3f}' for g in gaps]}")
        print(f"  Optimal k by gap: {optimal_k} (gap: {gaps[optimal_idx]:.3f})")

        return optimal_k

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
        elif method == 'DTW-Shape':
            return self._dtw_shape_clustering(data, n_clusters)
        elif method == 'Correlation':
            return self._correlation_clustering(data, n_clusters)
        elif method == 'Derivative':
            return self._derivative_clustering(data, n_clusters)
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

    def _normalize_shape(self, curve):
        """
        Normalize a time series to focus on shape rather than amplitude

        Methods applied:
        1. Z-score normalization (zero mean, unit variance)
        2. Min-max scaling to [0, 1] range

        Parameters:
        -----------
        curve : array
            Input time series

        Returns:
        --------
        normalized : array
            Shape-normalized time series
        """
        curve = np.array(curve, dtype=float)

        # Remove mean and scale to unit variance (z-score)
        mean = np.mean(curve)
        std = np.std(curve)
        if std > 0:
            normalized = (curve - mean) / std
        else:
            normalized = curve - mean

        # Additional min-max scaling to [0, 1]
        min_val = np.min(normalized)
        max_val = np.max(normalized)
        if max_val > min_val:
            normalized = (normalized - min_val) / (max_val - min_val)

        return normalized

    def _dtw_shape_clustering(self, data, n_clusters):
        """
        DTW-based clustering with shape normalization
        This reduces sensitivity to amplitude and focuses on shape
        """
        if not DTW_AVAILABLE:
            print("DTW library not available, using correlation clustering instead")
            return self._correlation_clustering(data, n_clusters)

        n_sources = len(data)

        # Normalize each curve to focus on shape
        print("  Normalizing curves for shape-based comparison...")
        data_normalized = np.array([self._normalize_shape(curve) for curve in data])

        # Calculate DTW distance matrix on normalized data
        dist_matrix = np.zeros((n_sources, n_sources))
        for i in range(n_sources):
            for j in range(i+1, n_sources):
                distance = dtw.distance(data_normalized[i], data_normalized[j])
                dist_matrix[i, j] = distance
                dist_matrix[j, i] = distance

        # Perform hierarchical clustering on distance matrix
        linkage_matrix = linkage(dist_matrix, method='average')
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1

        # Generate colors
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        return labels, colors

    def _correlation_clustering(self, data, n_clusters):
        """
        Correlation-based shape clustering
        Uses Pearson correlation as similarity measure (shape-focused)
        """
        n_sources = len(data)

        # Calculate correlation distance matrix
        # Correlation distance = 1 - abs(correlation)
        dist_matrix = np.zeros((n_sources, n_sources))

        for i in range(n_sources):
            for j in range(i+1, n_sources):
                # Pearson correlation coefficient
                corr = np.corrcoef(data[i], data[j])[0, 1]
                # Use 1 - |correlation| as distance (focuses on shape similarity)
                distance = 1 - np.abs(corr)
                dist_matrix[i, j] = distance
                dist_matrix[j, i] = distance

        # Perform hierarchical clustering
        linkage_matrix = linkage(dist_matrix, method='average')
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1

        # Generate colors
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        return labels, colors

    def _derivative_clustering(self, data, n_clusters):
        """
        Derivative-based shape clustering
        Clusters based on rate of change patterns (shape dynamics)
        """
        n_sources = len(data)

        # Calculate derivatives (rate of change) for each curve
        derivatives = []
        for curve in data:
            # First derivative (rate of change)
            deriv = np.diff(curve)
            # Normalize derivative
            if np.std(deriv) > 0:
                deriv = (deriv - np.mean(deriv)) / np.std(deriv)
            derivatives.append(deriv)

        derivatives = np.array(derivatives)

        # Standardize derivative data
        scaler = StandardScaler()
        derivatives_scaled = scaler.fit_transform(derivatives)

        # Perform hierarchical clustering on derivatives
        linkage_matrix = linkage(derivatives_scaled, method='ward')
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
        self.current_dataset_index = -1  # -1 for DSS, 0+ for datasets
        self.current_analysis_method = 'None'
        self.n_clusters = 3
        self.auto_clusters = False  # Whether to auto-determine cluster number
        self.cluster_method = 'silhouette'  # Method for auto-clustering
        self.selected_source = None
        self.cluster_labels = {}  # Store cluster labels for each tile: {tile: labels}
        self.cluster_colors = {}  # Store cluster colors for each tile: {tile: colors}
        self.cluster_sources = {}  # Store source coordinates for each tile: {tile: [(x, y), ...]}

        # Cluster cache directory
        self.output_dir = Path(output_dir)
        self.cluster_cache_dir = self.output_dir / 'cluster_cache'
        self.cluster_cache_dir.mkdir(exist_ok=True)

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
        ax_button = plt.axes([0.30, 0.01, 0.12, 0.04])
        self.btn_all = Button(ax_button, 'Show All Sources')
        self.btn_all.on_clicked(self.show_all_sources)

        # Add buttons for adjusting cluster number
        ax_button_minus = plt.axes([0.44, 0.01, 0.04, 0.04])
        self.btn_minus = Button(ax_button_minus, '-')
        self.btn_minus.on_clicked(self.decrease_clusters)

        ax_button_plus = plt.axes([0.49, 0.01, 0.04, 0.04])
        self.btn_plus = Button(ax_button_plus, '+')
        self.btn_plus.on_clicked(self.increase_clusters)

        # Add auto cluster button
        ax_button_auto = plt.axes([0.54, 0.01, 0.08, 0.04])
        self.btn_auto = Button(ax_button_auto, 'Auto')
        self.btn_auto.on_clicked(self.toggle_auto_clusters)

        # Add text to show current cluster number
        self.cluster_text = self.fig.text(0.70, 0.03, f'Clusters: {self.n_clusters}',
                                          fontsize=10, ha='left')

        # Add quick cluster button
        ax_button_cluster = plt.axes([0.63, 0.01, 0.10, 0.04])
        self.btn_cluster = Button(ax_button_cluster, 'Cluster (5)')
        self.btn_cluster.on_clicked(self.quick_cluster)

        # Add previous/next image buttons
        ax_button_prev = plt.axes([0.80, 0.01, 0.08, 0.04])
        self.btn_prev = Button(ax_button_prev, 'Prev')
        self.btn_prev.on_clicked(self.previous_image)

        ax_button_next = plt.axes([0.89, 0.01, 0.08, 0.04])
        self.btn_next = Button(ax_button_next, 'Next')
        self.btn_next.on_clicked(self.next_image)


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

                # Check if we have cluster colors for this tile
                if (self.current_tile in self.cluster_labels and
                    self.current_tile in self.cluster_colors and
                    self.current_tile in self.cluster_sources):
                    # Use cluster colors
                    labels = self.cluster_labels[self.current_tile]
                    colors = self.cluster_colors[self.current_tile]
                    cluster_coords = self.cluster_sources[self.current_tile]

                    # Create a mapping from coordinates to cluster labels
                    coord_to_label = {}
                    for idx, (cx, cy) in enumerate(cluster_coords):
                        coord_to_label[(cx, cy)] = labels[idx]

                    # Plot each source with its cluster color
                    for x, y in zip(x_coords, y_coords):
                        # Find matching cluster label (with tolerance for coordinate differences)
                        matched_label = None
                        min_dist = float('inf')

                        for (cx, cy), label in coord_to_label.items():
                            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                            if dist < min_dist and dist < 5.0:  # Within 5 pixels
                                min_dist = dist
                                matched_label = label

                        if matched_label is not None:
                            color = colors[matched_label]
                            self.ax_image.scatter(x, y, c=[color], s=50, alpha=0.7,
                                                marker='o', facecolors='none', edgecolors=[color], linewidths=2)
                        else:
                            # Source not in clustering (fallback to red)
                            self.ax_image.scatter(x, y, c='red', s=50, alpha=0.5,
                                                marker='o', facecolors='none', edgecolors='red', linewidths=1)
                else:
                    # Default red color
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
            self.current_dataset_index = -1
            print(f"\nSwitched to DSS reference image")
        else:
            self.current_image_type = label
            self.current_dataset_index = self.datasets.index(label)
            print(f"\nSwitched to {label} image")

        # Update display
        self.display_reference_image()

    def previous_image(self, event):
        """Switch to previous image in the sequence"""
        # Calculate previous index
        new_index = self.current_dataset_index - 1

        # Wrap around: if at DSS (-1), go to last dataset
        if new_index < -1:
            new_index = len(self.datasets) - 1

        # Update current image
        if new_index == -1:
            self.current_image_type = 'dss'
            self.current_dataset_index = -1
            print(f"\nSwitched to DSS reference image")
        else:
            self.current_image_type = self.datasets[new_index]
            self.current_dataset_index = new_index
            print(f"\nSwitched to {self.current_image_type} image")

        # Update radio button selection
        if new_index == -1:
            self.radio_img.set_active(0)
        else:
            self.radio_img.set_active(new_index + 1)

        # Update display
        self.display_reference_image()

    def next_image(self, event):
        """Switch to next image in the sequence"""
        # Calculate next index
        new_index = self.current_dataset_index + 1

        # Wrap around: if past last dataset, go to DSS
        if new_index >= len(self.datasets):
            new_index = -1

        # Update current image
        if new_index == -1:
            self.current_image_type = 'dss'
            self.current_dataset_index = -1
            print(f"\nSwitched to DSS reference image")
        else:
            self.current_image_type = self.datasets[new_index]
            self.current_dataset_index = new_index
            print(f"\nSwitched to {self.current_image_type} image")

        # Update radio button selection
        if new_index == -1:
            self.radio_img.set_active(0)
        else:
            self.radio_img.set_active(new_index + 1)

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
            self.auto_clusters = False
            self.n_clusters -= 1
            self.cluster_text.set_text(f'Clusters: {self.n_clusters}')
            self.fig.canvas.draw()
            print(f"\nCluster number decreased to: {self.n_clusters}")

    def increase_clusters(self, event):
        """Increase number of clusters"""
        if self.n_clusters < 10:
            self.auto_clusters = False
            self.n_clusters += 1
            self.cluster_text.set_text(f'Clusters: {self.n_clusters}')
            self.fig.canvas.draw()
            print(f"\nCluster number increased to: {self.n_clusters}")

    def toggle_auto_clusters(self, event):
        """Toggle automatic cluster number determination"""
        self.auto_clusters = not self.auto_clusters
        if self.auto_clusters:
            self.cluster_text.set_text(f'Clusters: Auto')
            print(f"\nAuto cluster mode enabled (using {self.cluster_method} method)")
        else:
            self.cluster_text.set_text(f'Clusters: {self.n_clusters}')
            print(f"\nAuto cluster mode disabled (using {self.n_clusters} clusters)")
        self.fig.canvas.draw()

    def _get_cluster_cache_path(self, tile):
        """Get the cache file path for a tile's cluster results"""
        return self.cluster_cache_dir / f'{tile}_cluster5_dtw_shape.pkl'

    def _save_cluster_results(self, tile, labels, colors, source_coords):
        """Save cluster results to cache"""
        cache_file = self._get_cluster_cache_path(tile)
        cache_data = {
            'labels': labels,
            'colors': colors,
            'source_coords': source_coords,
            'method': 'DTW-Shape',
            'n_clusters': 5
        }
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Cluster results saved to: {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cluster results: {e}")

    def _load_cluster_results(self, tile):
        """Load cluster results from cache if available"""
        cache_file = self._get_cluster_cache_path(tile)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                return cache_data['labels'], cache_data['colors'], cache_data['source_coords']
            except Exception as e:
                print(f"Warning: Failed to load cluster results: {e}")
                return None, None, None
        return None, None, None

    def quick_cluster(self, event):
        """Quick clustering with 5 clusters using DTW-Shape method"""
        if not self.current_tile:
            print("\nNo tile selected!")
            return

        print("\n" + "="*80)
        print("Quick Clustering Analysis (5 clusters, DTW-Shape method)")
        print("="*80)

        # Try to load cached results first
        cached_labels, cached_colors, cached_coords = self._load_cluster_results(self.current_tile)
        if cached_labels is not None:
            print(f"Loading cached cluster results for tile {self.current_tile}...")
            self.cluster_labels[self.current_tile] = cached_labels
            self.cluster_colors[self.current_tile] = cached_colors
            self.cluster_sources[self.current_tile] = cached_coords

            # Print distribution (sorted by size)
            cluster_sizes = []
            for cluster_id in range(5):
                count = np.sum(cached_labels == cluster_id)
                cluster_sizes.append((cluster_id, count))
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)

            print(f"\nCluster distribution (sorted by size):")
            for rank, (cluster_id, count) in enumerate(cluster_sizes):
                percentage = (count / len(cached_labels)) * 100
                color_name = ['Gray', 'Light-Gray', 'Red', 'Green', 'Yellow'][rank]
                print(f"  Cluster {cluster_id}: {count} sources ({percentage:.1f}%) - {color_name}")

            print(f"\nImage markers updated with cluster colors!")
            print("="*80)

            # Update display
            self.display_reference_image()
            return

        # Get all light curves for current tile
        light_curves = []
        source_coords = []  # Store (x, y) coordinates

        first_dataset = sorted(self.phot_data.datasets.keys())[0]
        if first_dataset not in self.phot_data.datasets:
            print("No data available!")
            return

        if self.current_tile not in self.phot_data.datasets[first_dataset]:
            print(f"No data for tile {self.current_tile}!")
            return

        sources = self.phot_data.datasets[first_dataset][self.current_tile]

        # Get expected number of time points
        n_datasets = len(self.phot_data.datasets)
        dataset_names = sorted(self.phot_data.datasets.keys())

        for i, source in enumerate(sources):
            light_curve = self.phot_data.get_light_curve(source['x'], source['y'], self.current_tile)

            # Create a flux array with zeros for missing data
            fluxes = []
            lc_dict = {lc['dataset']: lc['flux'] for lc in light_curve}

            for dataset_name in dataset_names:
                if dataset_name in lc_dict:
                    fluxes.append(lc_dict[dataset_name])
                else:
                    # Fill missing data with 0
                    fluxes.append(0.0)

            # Include all sources (with zero-padding if needed)
            if len(light_curve) >= 1:  # At least one measurement
                light_curves.append(fluxes)
                source_coords.append((source['x'], source['y']))

        if len(light_curves) < 5:
            print(f"Not enough sources (found {len(light_curves)}, need at least 5)")
            return

        # Count complete vs padded sources
        complete_count = sum(1 for lc in light_curves if all(f != 0 for f in lc))
        padded_count = len(light_curves) - complete_count

        print(f"Analyzing {len(light_curves)} sources with {n_datasets} time points each...")
        print(f"  Complete light curves: {complete_count}")
        print(f"  Zero-padded light curves: {padded_count}")

        # Perform clustering with 5 clusters using DTW-Shape method
        try:
            labels, colors = self.trend_analyzer.analyze(light_curves, method='DTW-Shape', n_clusters=5)

            # Reassign colors based on cluster size (large clusters get dim colors, small get bright)
            cluster_sizes = []
            for cluster_id in range(5):
                count = np.sum(labels == cluster_id)
                cluster_sizes.append((cluster_id, count))

            # Sort by size (descending)
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)

            # Define fixed color scheme: dim colors for large clusters, bright for small
            fixed_colors = [
                [0.7, 0.7, 0.7, 1.0],    # Gray (for largest cluster)
                [0.8, 0.8, 0.6, 1.0],    # Light yellow-gray (for 2nd largest)
                [1.0, 0.0, 0.0, 1.0],    # Red (for 3rd)
                [0.0, 1.0, 0.0, 1.0],    # Green (for 4th)
                [1.0, 1.0, 0.0, 1.0]     # Yellow (for smallest)
            ]

            # Create mapping from old cluster ID to new color
            new_colors = np.zeros((5, 4))
            for rank, (cluster_id, count) in enumerate(cluster_sizes):
                new_colors[cluster_id] = fixed_colors[rank]

            # Store results with source coordinates
            self.cluster_labels[self.current_tile] = labels
            self.cluster_colors[self.current_tile] = new_colors
            self.cluster_sources[self.current_tile] = source_coords

            # Save results to cache
            self._save_cluster_results(self.current_tile, labels, new_colors, source_coords)

            # Print results (sorted by size)
            print(f"\nClustering complete!")
            print(f"Cluster distribution (sorted by size):")
            for rank, (cluster_id, count) in enumerate(cluster_sizes):
                percentage = (count / len(labels)) * 100
                color_name = ['Gray', 'Light-Gray', 'Red', 'Green', 'Yellow'][rank]
                print(f"  Cluster {cluster_id}: {count} sources ({percentage:.1f}%) - {color_name}")

            # Update display
            self.display_reference_image()

            print("\nImage markers updated with cluster colors!")
            print("="*80)

        except Exception as e:
            print(f"Error during clustering: {str(e)}")
            import traceback
            traceback.print_exc()

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

        # Determine number of clusters
        if self.auto_clusters:
            print(f"\nAuto-determining optimal number of clusters using {self.cluster_method} method...")
            optimal_k = self.trend_analyzer.estimate_optimal_clusters(
                all_light_curves,
                method=self.cluster_method,
                max_clusters=min(10, n_plotted - 1)
            )
            actual_n_clusters = optimal_k
            print(f"  Using {actual_n_clusters} clusters")
        else:
            actual_n_clusters = self.n_clusters
            print(f"\nUsing manual cluster setting: {actual_n_clusters} clusters")

        # Perform trend analysis
        print(f"\nPerforming {self.current_analysis_method} analysis...")
        labels, colors = self.trend_analyzer.analyze(
            all_light_curves,
            method=self.current_analysis_method,
            n_clusters=actual_n_clusters
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
        if self.auto_clusters:
            title += f'{n_plotted} sources, {n_classes} classes (auto)'
        else:
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

