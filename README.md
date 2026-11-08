# Vectorized k-means

A lightweight k-means implementation with vectorized distance calculations, cluster assignments, and centroid updates.

## Features

- Pure Python/NumPy implementation.
- A `KMeans` class that manages configuration, scaling, and fitted centroids.
- A standalone `run_kmeans()` function for clustering inputs as provided.
- Early stopping when cluster assignments are unchanged and no clusters are empty.
- Configurable cluster count, max iterations, and random seed.

## Example Usage

```python
import numpy as np
from kmeans import KMeans

X = np.random.randn(100, 5) # simulate training data
kmeans_model = KMeans(5)
clusters = kmeans_model.fit_predict(X)

X_new = np.random.randn(1, 5) # simulate new incoming data to assign a cluster to
cluster = kmeans_model.predict(X_new)

```

## Notes

`KMeans` applies min-max scaling to the inputs automatically and reuses those scaling parameters for prediction. To manage preprocessing yourself, call `run_kmeans()` directly. It returns cluster assignments and centroids without scaling the inputs.
