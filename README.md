# Unsupervised Learning for Sea Ice and Lead Classification

## Overview
This repository contains code for applying unsupervised learning techniques to Earth Observation (EO) data for sea ice and lead discrimination. The project focuses on two main tasks:
1. Discrimination of sea ice and leads using Sentinel-2 optical data
2. Discrimination of sea ice and leads using Sentinel-3 altimetry data

## Background
Sea ice monitoring is critical for climate research, maritime navigation, and Arctic ecosystem studies. This project demonstrates how unsupervised machine learning techniques can effectively classify sea ice features without labeled training data.

## Methods
We implement and compare two primary unsupervised learning approaches:

### K-means Clustering
K-means partitions data into k clusters by minimizing within-cluster variance. The algorithm iteratively:
- Assigns data points to the nearest centroid
- Recalculates centroids based on assigned points
- Repeats until convergence

### Gaussian Mixture Models (GMM)
GMMs model data as a mixture of several Gaussian distributions. Unlike K-means, GMMs provide:
- Soft clustering with probability assignments
- Flexibility in cluster covariance (size/shape)
- Better handling of overlapping distributions

## Dataset
The project uses two primary datasets:
- **Sentinel-2 optical imagery** (bands B1, B2, B3)
- **Sentinel-3 altimetry data** with derived features:
  - Waveform peakiness (PP)
  - Stack standard deviation (SSD)
  - Backscatter coefficient (sigma0)

## Implementation Details

### Image Classification (Sentinel-2)
The code includes implementations for:
- Reading and preprocessing satellite imagery
- Applying K-means and GMM clustering to spectral bands
- Visualizing classification results

### Altimetry Classification (Sentinel-3)
The altimetry analysis includes:
- Data extraction from NetCDF files
- Feature engineering (peakiness, SSD calculation)
- Model training and evaluation
- Comparison with ESA reference data

## Key Functions

### Peakiness Calculation
```python
def peakiness(waves, **kwargs):
    """Calculates the peakiness of radar waveforms."""
    # Implementation details in code
```

### Stack Standard Deviation (SSD)
```python
def calculate_SSD(RIP):
    """Calculates Stack Standard Deviation from Range Integrated Power (RIP)."""
    # Implementation details in code
```

### Data Loading
```python
def unpack_gpod(variable):
    """Extracts and preprocesses variables from GPOD NetCDF files."""
    # Implementation details in code
```

## Results
The unsupervised classification achieves excellent agreement with ESA reference data:
- Precision: ~99%
- Recall: ~99% 
- F1-score: ~99%
Classification Report:
              precision    recall  f1-score   support
         0.0       1.00      1.00      1.00      8878
         1.0       0.99      0.99      0.99      3317
    accuracy                           1.00     12195
### Confusion Matrix
```
[8856   22]
[  24 3293]
```

## Visualizations
The repository includes code for visualizing:
- Cluster assignments on satellite imagery
- Mean waveform profiles for each class
- Feature space plots showing cluster separation
- Aligned waveforms using cross-correlation

## Requirements
- Python 3.x
- NumPy
- scikit-learn
- matplotlib
- rasterio
- netCDF4
- SciPy

## Usage

### Sentinel-2 Analysis
```python
# K-means on Sentinel-2 data
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_
```

### Sentinel-3 Analysis
```python
# Gaussian Mixture Model on altimetry features
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data_cleaned)
clusters_gmm = gmm.predict(data_cleaned)
```

## Future Work
- Integration of supervised and unsupervised approaches
- Extension to time series analysis for monitoring sea ice changes
- Exploration of additional features and clustering algorithms

## Citation
If you use this code in your research, please cite:
```
@misc{SeaIce and Lead UnsupervisedLearning,
  author = {WeiFu},
  title = {Unsupervised Learning for Sea Ice and Lead Classification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/WeiWeiiFu/1.git}
}
```

## License
[MIT License](LICENSE)
