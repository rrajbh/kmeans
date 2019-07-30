#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:37:59 2019

@author: rajat.rajbhandari
"""

#k-means clustering ith expectaion -maximization
import numpy as np
import matplotlib.pyplot as plt 


from sklearn.datasets.samples_generator import make_blobs

X,y_true = make_blobs(n_samples=300,centers=4,cluster_std=0.60, random_state=0)
plt.scatter(X[:,0], X[:,1],s=50)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=4)
km.fit(X)
y_km=km.predict(X)

#plot data points in different colors along with their centers

plt.scatter(X[:, 0], X[:, 1], c=y_km, s=50, cmap='viridis')
centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

from sklearn_metrics import pairwise_distances_argmin

def find_clusters (X,n_clusters, rseed=2):
    #1. randomly choose clusters
    rng = np.random.RandomState(rseed)
    i=rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        #2a. assign labels based on closest center
        labels = pairwise_distances_argmin(X,centers)
        
        #2b. find new centers from means of points
        new_centers = np.array([X[labels ==i].mean(0)
                                for i in range(n_clusters)])
        
        #2c check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels
    
    
    
centers, labels = find_clusters(X,4)
plt.scatter (X[:,0],X[:,1], c=labels, s=50, cmap='virdis')
    
    








