############################################
#EM algorithm for a Guassian mixture model #
############################################

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

def gaussian_mixture_model(data):
    """
    EM algorithm for a Guassian mixture model with 2 gaussians on the given dataset, and returns the parameters for the model.
    
    :param np.ndarray data: The data of size (n, k), where n is the number of points and k is the dimensionality of each point.
    :rtype tuple: (pi_1, pi_2, mu_1, mu_2, sigma_1, sigma_2)
    """

    n = data.shape[0]  # number of points
    k = data.shape[1]  # the dimensionality of each point
    num_iterations = 80

    # initialize the priors to be equal
    pi_1 = 0.5
    pi_2 = 0.5

    # get the initial means as random datapoints
    initial_means = rnd.permutation(data)[:2]
    mu_1 = initial_means[0, :]
    mu_2 = initial_means[1, :]

    # start with the covariance matrices as the identity matrix times a large constant
    sigma_1 = np.identity(k) * 100
    sigma_2 = np.identity(k) * 100

    z = np.zeros((n, 2))
    for iter in range(num_iterations):
        # E-step
        pdf_1 = multivariate_normal(mean=mu_1, cov=sigma_1)
        pdf_2 = multivariate_normal(mean=mu_2, cov=sigma_2)
        for i in range(n):
            # prob_1 = pi_1 * pdf(data[i, :], mu_1, sigma_1)
            prob_1 = pi_1 * pdf_1.pdf(data[i, :])
            # prob_2 = pi_2 * pdf(data[i, :], mu_2, sigma_2)
            prob_2 = pi_2 * pdf_2.pdf(data[i, :])
            z[i, 0] = prob_1 / (prob_1 + prob_2)
            z[i, 1] = prob_2 / (prob_1 + prob_2)

        # M-step
        # Update pi
        sums = np.sum(z, axis=0)
        pi_1 = sums[0] / n
        pi_2 = sums[1] / n

        # Update mu
        new_mu_1 = np.zeros(k)
        new_mu_2 = np.zeros(k)
        for i in range(n):
            new_mu_1 += z[i, 0] * data[i, :]
            new_mu_2 += z[i, 1] * data[i, :]
        mu_1 = new_mu_1 / sums[0]
        mu_2 = new_mu_2 / sums[1]

        # Update sigma
        new_sigma_1 = np.zeros((k, k))
        new_sigma_2 = np.zeros((k, k))
        for i in range(n):
            tmp = np.zeros((k, 1))
            tmp[:, 0] = data[i, :] - mu_1
            new_sigma_1 += z[i, 0] * np.dot(tmp, tmp.T)
            tmp[:, 0] = data[i, :] - mu_2
            new_sigma_2 += z[i, 1] * np.dot(tmp, tmp.T)
        sigma_1 = new_sigma_1 / sums[0]
        sigma_2 = new_sigma_2 / sums[1]

    return pi_1, pi_2, mu_1, mu_2, sigma_1, sigma_2

def dist(x, centroid):
    res = 0
    for i in range(len(x)):
        res += (x[i] - centroid[i])**2
    return res


#####################################################################################################################
# Runs the K-means algorithm to find two clusters on the given dataset, and returns the centroids for each cluster. #
#####################################################################################################################

def kmeans(data):
    """
    Runs the K-means algorithm
    
    :param np.ndarray data: The data of size (n, k), where n is the number of points and k is the dimensionality of
    each point.
    :rtype tuple: (centroid_1, centroid_2)
    """
    def dist(x, centroid):
        res = 0
        for i in range(len(x)):
            res += (x[i] - centroid[i])**2
        return res

    n = data.shape[0]  # number of points
    k = data.shape[1]
    num_iterations = 80

    # get the initial means as random datapoints
    initial_means = rnd.permutation(data)[:2]
    centroid_1 = initial_means[0, :]
    centroid_2 = initial_means[1, :]

    z = np.zeros(n)
    for iter in range(num_iterations):
        # E-step
        for i in range(n):
            dist_1 = dist(data[i, :], centroid_1)
            dist_2 = dist(data[i, :], centroid_2)
            if dist_1 < dist_2:
                z[i] = 0
            else:
                z[i] = 1

        # M-step
        new_centroid_1 = np.zeros(k)
        cnt_1 = 0
        new_centroid_2 = np.zeros(k)
        cnt_2 = 0
        for i in range(n):
            if z[i] == 0:
                new_centroid_1 += data[i, :]
                cnt_1 += 1
            else:
                new_centroid_2 += data[i, :]
                cnt_2 += 1
        centroid_1 = new_centroid_1 / cnt_1
        centroid_2 = new_centroid_2 / cnt_2

    return centroid_1, centroid_2

# Run the GMM
pi_1, pi_2, mu_1, mu_2, sigma_1, sigma_2 = gaussian_mixture_model(input_data)

# Run K-means
centroid_1, centroid_2 = kmeans(input_data)

# Make the hard cluster assignments and plot the data
gmm_clusters = np.zeros(input_data.shape[0])
kmeans_clusters = np.zeros(input_data.shape[0])

pdf_1 = multivariate_normal(mean=mu_1, cov=sigma_1)
pdf_2 = multivariate_normal(mean=mu_2, cov=sigma_2)

for i in range(input_data.shape[0]):
    # GMM
    prob_1 = pi_1 * pdf_1.pdf(input_data[i, :])
    prob_2 = pi_2 * pdf_2.pdf(input_data[i, :])
    gmm_clusters[i] = 0 if prob_1 > prob_2 else 1

    # K-means
    dist_1 = dist(input_data[i, :], centroid_1)
    dist_2 = dist(input_data[i, :], centroid_2)
    kmeans_clusters[i] = 0 if dist_1 < dist_2 else 1

# Plot GMM
plt.figure(1)
plt.title('Cluster assignments using MoG')
colors = ['b' if cluster == 0 else 'r' for cluster in gmm_clusters]
plt.scatter(input_data[:, 0], input_data[:, 1], c=colors)
    
# Plot K-means
plt.figure(2)
plt.title('Cluster assignments using K-means')
colors = ['b' if cluster == 0 else 'r' for cluster in kmeans_clusters]
plt.scatter(input_data[:, 0], input_data[:, 1], c=colors)
