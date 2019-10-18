from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from numpy import cov
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

seed(1)

# 1. (a) Generate a random artificial data set that shows nonlinear correlations. Ideal (target) covariance cov(X, Y) = 0, yet X and Y are not independent and feature dependencies of your choice.

N = 5
for n in range(1, N)
Z = randn(1)
X = Z * 10 + randn(1)
Y = Z ** 2 + randn(1)

# Can your ideal target covariance of zero be reached? If not, why not?

print('No, because there will always be some variance, even if the covariance is perfectly negative.')

# 1. (b) Create surrogates exhibiting a given correlation coefficient r=r_{xy} (parameter of the function). Create target examples for r=1, r=0, r=0.5, r=-0.5 and r=-1. Decide yourself which plots you want to present and are meaningful.

def create_surrogates_with_given_correlation(correlation_coefficient, sample_size):


# 1. (c) Implement a causal relationship of the common effect case. Compute the correlations (in terms of r) between X and Y, and X and Z.

# 1. (d) Optional. Study numerically how the sample variance of the sample mean of n samples of a random variable with target \mu and target \sigma^2 depends on the sample size n. Target refers to the mean and variance of the ideal random variable (not the realized sample).
# Is it var(\bar{X}) = \sigma^2 / n?


# 2. PCA: Create a surrogate data set for the cases (a, 4 blobs) and (b, 2 touching parabola spreads) as shown in the lecture, but in a higher-dimensional space (not 2d). Perform a PCA/Class prediction with ovr logistic regression analysis as developed in the lecture. Study prediction boundaries.


# 3. K-Means (a) Create surrogate data in 2 dimension. Create 4 blobs (clusters), labeled. Perform k-means analysis as shown in lecture. Design data such that the 4 blobs are not overlapping.

# 3. K-Means (b) Design data such that the 4 blobs are partially overlapping. Compare the elbow plots of (a) and (b). Details are given in class.

# 3. K-Means (c) Optional. Study more complicated cases, find or develop quantitative measures.

