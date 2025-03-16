import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def generate_data(means, covariances, priors, N=1000):
    labels = np.random.choice(len(means), size=N, p=priors)
    data = np.vstack([np.random.multivariate_normal(means[i], covariances[i], (labels == i).sum())
                      for i in range(len(means))])
    true_labels = np.hstack([np.full((labels == i).sum(), i) for i in range(len(means))])
    return data, true_labels

def euclidean_classifier(x, means):
    distances = [np.linalg.norm(x - mean) for mean in means]
    return np.argmin(distances)

def mahalanobis_classifier(x, means, covariances):
    cov_invs = [np.linalg.inv(cov) for cov in covariances]
    distances = [np.sqrt((x - mean).T @ cov_inv @ (x - mean)) for mean, cov_inv in zip(means, cov_invs)]
    return np.argmin(distances)

def bayesian_classifier(x, means, covariances, priors):
    probs = [multivariate_normal.pdf(x, mean=means[i], cov=covariances[i]) * priors[i] 
             for i in range(len(means))]
    return np.argmax(probs)

def classify_and_evaluate(data, true_labels, classifier, *args):
    predicted_labels = np.array([classifier(x, *args) for x in data])
    error = np.mean(predicted_labels != true_labels)
    return predicted_labels, error

def plot_decision_boundaries(means, covariances, priors, dataset, true_labels, classifier, title):
    x_min, x_max = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
    y_min, y_max = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    if classifier == bayesian_classifier:
        predictions = np.array([classifier(p, means, covariances, priors) for p in grid_points])
    elif classifier == euclidean_classifier:
        predictions = np.array([classifier(p, means) for p in grid_points])
    elif classifier == mahalanobis_classifier:
        predictions = np.array([classifier(p, means, covariances) for p in grid_points])
    
    predictions = predictions.reshape(xx.shape)
    
    plt.contourf(xx, yy, predictions, alpha=0.3, cmap='viridis')
    plt.scatter(dataset[:, 0], dataset[:, 1], c=true_labels, cmap='viridis', edgecolors='k')
    plt.title(title)
    plt.show()