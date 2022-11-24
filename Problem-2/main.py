
from kmeans import *



if __name__ == "__main__":
    with open("./data/seeds.txt") as textFile:
        listTo2D = [line.split() for line in textFile]

    data_seeds = np.array(listTo2D)
    data_seeds = data_seeds.astype(np.float64)
    # define some metrics 
    total_iters = 10
    given_clusters = [3, 5, 7]
    cluster_each_mean = [[0] * 0 for i in range(len(given_clusters))]
    # loop for len of given_clusters
    for clusters_given in range(len(given_clusters)):
        # loop for 10 iterartions and get SEE
        for i in range(total_iters):
            SSE_forEach = kmeans(data_seeds, given_clusters[clusters_given])
            cluster_each_mean[clusters_given].append(SSE_forEach)
    total_mean = np.array(np.mean(cluster_each_mean, axis=1))
    for i in range(len(total_mean)):
        print('Number of clusters is k =',given_clusters[i],'and SSE average of', total_iters, 'iterations is:',  total_mean[i])