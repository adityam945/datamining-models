import numpy as np
import random

def euclideanDitance(x, y):
    # eculedian distacnce 
    distance = np.linalg.norm(x - y)
    return distance

def createClusters(number_clusters, centroids, data_seeds):
    # creating clusters
    clusters = [[0] * 0 for i in range(number_clusters)]
    for data_index in range(data_seeds.shape[0]):
        dist_np = []
        for i in range(number_clusters):
            distance = euclideanDitance(centroids[i], data_seeds[data_index])
            dist_np.append(distance)
        min_index = np.argmin(dist_np)
        clusters[min_index].append(data_seeds[data_index])
    return clusters

def updateCentroid(number_clusters, centroids, data_seeds):
    # update centroids
    updatedClutsters = createClusters(number_clusters, centroids, data_seeds)
    centriods = []
    for i in range(number_clusters):
        cluster_each = np.array(updatedClutsters[i])
        cluster_each_transpose = cluster_each.T
        new_centroid = np.mean(cluster_each_transpose, axis=1)
        centriods.append(new_centroid)
    return centriods, updatedClutsters

def calculateSSE(clusters, centroids, number_clusters):
    # cal SSE using 
    distance_sum_each_cluster = []
    for i in range(number_clusters):
        distance_sq_each_cluster = []
        for j in range(len(clusters[i])):
            distace = euclideanDitance(centroids[i], clusters[i][j])
            distance_sq_each_cluster.append(distace * distace)
        distance_sum_each_cluster.append(sum(distance_sq_each_cluster))
        distance_sq_each_cluster = []
    return sum(distance_sum_each_cluster)

def kmeans(data_seeds, number_clusters):
    # loop for 100 iterations or until SSE is not less than 0.001 for 2 successive iters
    calculated_SSE = None
    i = 0
    SSE = []
    SSE_True = True
    centroids = data_seeds[np.random.choice(data_seeds.shape[0], size=number_clusters, replace=False)]
    # begin loop
    while i < 100 and SSE_True:
        centroids, clusters = updateCentroid(number_clusters, centroids, data_seeds)
        # break if SSE conditions statisfy else append
        calculated_SSE = calculateSSE(clusters, centroids, number_clusters)
        if len(SSE) < 2:
            SSE.append(calculated_SSE)
        else:
            SSE.append(calculated_SSE)
            pre_penultimate = SSE[i - 2]
            penultimate = SSE[i - 1] 
            current_SSE = SSE[i]
            sub_penultimate =  abs(current_SSE - penultimate )
            sub_pre_penultimate =   abs(current_SSE - pre_penultimate)
            # 
            if sub_penultimate < 0.001 and sub_pre_penultimate < 0.001:
                # break the clustering here
                SSE_True = False
                break
        # increase the iters or value of i
        i += 1
    return calculated_SSE