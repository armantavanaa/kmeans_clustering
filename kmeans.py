import numpy as np
import math


def select_centroids(X, k):
    """
    kmeans++ algorithm to select initial points:

    1. Pick first point randomly
    2. Pick next k-1 points by selecting points that maximize the minimum
       distance to all existing clusters. So for each point, compute distance
       to each cluster and find that minimum.  Among the min distances to a cluster
       for each point, find the max distance. The associated point is the new centroid.

    Return centroids as k x p array of points from X.
    """
    idx = np.random.choice(len(X), 1, replace=False)
    cur_centroids = list(X[idx])
    for i in range(k-1):
        dist = np.array([np.min([(c - x)**2 for c in cur_centroids]) for x in X])
        dist_over_n = dist / dist.sum()
        cur_centroids = np.concatenate((cur_centroids, [X[np.argmax(dist_over_n)]]))

    return cur_centroids

def kmeans(X: np.ndarray, k: int, centroids=None, max_iter=30, tolerance=1e-2):
    if centroids == "kmeans++":
        cur_centroids = select_centroids(X, k)
    else:
        idx = np.random.choice(len(X), k, replace=False)
        cur_centroids = X[idx]
    p = 0
    dist = math.inf
    labels = [[] for i in range(k)]
    while p < max_iter and dist >= tolerance:
        labels = [[] for i in range(k)]
        prev_centroids = cur_centroids
        for i in range(len(X)):
            j = np.argmin(np.linalg.norm(X[i] - prev_centroids, axis=1))
            labels[j].append(i)

        cur_centroids = np.array([np.mean(X[labels[j]], axis=0) for j in range(k)])
        dist = np.all(np.linalg.norm(cur_centroids - prev_centroids, axis=1))
        p += 1

    final_labels = [0 for i in range(len(X))]
    for i in range(len(labels)):
        for j in labels[i]:
            final_labels[j] = i

    labels = final_labels
    return cur_centroids, labels


