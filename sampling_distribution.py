import numpy as np

def sample_from_distribution(distribution: np.array, samples):
    norm_distr = distribution / np.linalg.norm(distribution, ord=1)
    cumsum = np.cumsum(norm_distr)
    rand = np.random.rand(samples)    
    return np.array([np.argmax(cumsum >= val) for val in rand])

def noisy_sample_from_distribution(distribution: np.array, samples):
    norm_distr = distribution / np.linalg.norm(distribution, ord=1)
    noise = np.random.rand(len(distribution)) * (1.0 / float(len(distribution) + 1))
    return sample_from_distribution(norm_distr + noise, samples)

def resample_from_s2_up(distribution: np.array) -> np.array:
    return distribution / 2.0 + 0.5

def resample_from_s2_down(distribution: np.array) -> np.array:
    return distribution / (-2.0) + 0.5
