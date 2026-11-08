import numpy as np


class KMeans:
    '''
    Wrapper class for run_kmeans() to facilitate state management 
    (e.g. configuration, scaling, fitted centroids).

    Applies min-max scaling during fitting and reuses the fitted 
    scaling parameters for prediction.
    '''

    def __init__(self, n_clusters, max_iter=300, random_seed=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.centroids = None
        self.min_ = None
        self.ranges = None

    def fit_predict(self, X):
        X = np.asarray(X)
        # Scale the inputs to (0, 1) to prevent large features from
        # dominating the distance computation. Set calc_params to
        # True so that the scaling parameters are up-to-date with
        # the fitted inputs.
        X = self.scale_inputs(X, calc_params=True)

        clusters, self.centroids = run_kmeans(
            X, self.n_clusters, max_iter=self.max_iter, random_seed=self.random_seed)

        return clusters

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("KMeans not fitted, call fit_predict() first.")
        if X.ndim < 2:
            raise ValueError("Inputs must be 2D.")

        X = np.asarray(X)
        X = self.scale_inputs(X)

        return assign_clusters(X, self.centroids)

    def scale_inputs(self, X, calc_params=False):
        if calc_params or self.min_ is None:
            self.min_ = np.min(X, axis=0)
            self.ranges = np.max(X, axis=0) - self.min_
            # Prevent division by zero error.
            self.ranges[self.ranges == 0] = 1

        return (X - self.min_) / self.ranges


def run_kmeans(X, k, max_iter=300, random_seed=42):
    '''
    Runs a vectorized version of k-means, assigning each row 
    of `X` to one of `k` clusters. 

    Args
    ----
    X: np.ndarray
        Numeric inputs with shape (num_rows, num_features).
    k: int
        Number of clusters.
    max_iter: int
        Maximum number of iterations for calculating  the clusters.
    random_seed: int or None
        Random seed for NumPy's random number generator to ensure 
        reproducible results across runs.

    Returns
    -------
    clusters: np.ndarray 
        Integer cluster assignment for each row, shape (num_rows,).
    centroids: np.ndarray
        Final centroid positions for each cluster with shape 
        (k, num_features).
    '''

    nrows, nfeatures = X.shape

    assert k > 0 and k < nrows

    # Initialize the centroids with randomly selected rows.
    rng = np.random.default_rng(seed=random_seed)
    centroids = X[rng.choice(nrows, size=k, replace=False)]
    prev_clusters = None
    nempty = 0
    # Initialize the clusters using the randomly selected
    # centroids.
    clusters = assign_clusters(X, centroids)

    for i in range(max_iter):

        # Stop if cluster assignments are unchanged and there are no
        # empty clusters.
        if prev_clusters is not None and nempty == 0 and np.array_equal(clusters, prev_clusters):
            break

        prev_clusters = clusters

        # Initialize per feature totals for each cluster
        clusters_sum = np.zeros((k, nfeatures))

        # np.add.at() adds values from `X` into `clusters_sum` at the
        # indices in `clusters`, including indices that are repeated.
        np.add.at(clusters_sum, clusters, X)

        # Get the number of rows assigned to each cluster.
        cluster_counts = np.bincount(clusters, minlength=k)

        # Check for empty clusters.
        empty = cluster_counts == 0
        nempty = sum(empty)

        # Prevent division by zero.
        if nempty > 0:
            cluster_counts[empty] = 1

        # Update the centroids using the mean of each cluster.
        new_centroids = clusters_sum / cluster_counts.reshape(-1, 1)

        # Reinitialize empty cluster centroids with randomly selected rows.
        if nempty > 0:
            new_centroids[empty] = X[rng.choice(nrows, size=nempty, replace=False)]

        centroids = new_centroids

        # Reassign rows using the updated centroids
        clusters = assign_clusters(X, centroids)

    return clusters, centroids


def assign_clusters(X, centroids):
    # Compute the squared Euclidean distance between every row
    # and every centroid. Broadcasting allows the difference between
    # the centriods and `X` to be calculated directly instead of
    # iterating over the rows. Omitting the square root doesn't
    # change which centroid is closest.
    distances = np.sum((X[:, np.newaxis, :] - centroids)**2, axis=-1)

    # Get the location of the smallest distance in each row (i.e.
    # the assigned cluster).
    return np.argmin(distances, axis=1)
