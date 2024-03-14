import numpy as np
import scipy.stats
import torch

def calculate_skewness_torch(x):
    mean = torch.mean(x)
    std_dev = torch.std(x)
    skewness = torch.mean((x - mean) ** 3) / (std_dev ** 3)
    return skewness

def calculate_kurtosis_torch(x):
    mean = torch.mean(x)
    std_dev = torch.std(x)
    kurtosis = torch.mean((x - mean) ** 4) / (std_dev ** 4) - 3
    return kurtosis

def calculate_fos_features(x):
    if isinstance(x, np.ndarray):
        energy = np.sum(x**2)
        total_energy = np.sum(x)
        entropy = scipy.stats.entropy(x)
        minimum = np.min(x)
        tenth_percentile = np.percentile(x, 10)
        ninetieth_percentile = np.percentile(x, 90)
        maximum = np.max(x)
        mean = np.mean(x)
        median = np.median(x)
        interquartile_range = scipy.stats.iqr(x)
        range_ = maximum - minimum
        mad = np.mean(np.abs(x - mean))
        rmad = np.median(np.abs(x - median))
        rms = np.sqrt(np.mean(x**2))
        std_dev = np.std(x)
        skewness = scipy.stats.skew(x)
        kurtosis = scipy.stats.kurtosis(x)
        variance = np.var(x)
        uniformity = len(np.unique(x)) / len(x)
    elif isinstance(x, torch.Tensor):
        energy = torch.sum(x**2).item()
        total_energy = torch.sum(x).item()
        #histogram of x
        h=torch.histogram(x.flatten(),256,density=True)
        # remove from h the zero values
        h=h[0][h[0]>1e-5]
        entropy = -torch.sum(h * torch.log(h)).item()
        minimum = torch.min(x).item()
        tenth_percentile = torch.quantile(x.flatten(), 0.1).item()
        ninetieth_percentile =torch.quantile(x.flatten(), 0.9).item()
        maximum = torch.max(x).item()
        mean = torch.mean(x).item()
        median = torch.median(x).item()
        interquartile_range = tenth_percentile - ninetieth_percentile
        range_ = maximum - minimum
        mad = torch.mean(torch.abs(x - mean)).item()
        rmad = torch.median(torch.abs(x - median)).item()
        rms = torch.sqrt(torch.mean(x**2)).item()
        std_dev = torch.std(x).item()
        variance = torch.var(x).item()
        skewness = calculate_skewness_torch(x).item()
        kurtosis = calculate_kurtosis_torch(x).item()
        unique_elements = torch.unique(x[0])
        uniformity = len(unique_elements) / len(x[0])
        
    cv = std_dev / mean

    return {
        'Energy': energy,
        'CV':cv,
        'Total Energy': total_energy,
        'Entropy': entropy,
        'Minimum': minimum,
        '10th Percentile': tenth_percentile,
        '90th Percentile': ninetieth_percentile,
        'Maximum': maximum,
        'Mean': mean,
        'Median': median,
        'Interquartile Range': interquartile_range,
        'Range': range_,
        'Mean Absolute Deviation': mad,
        'Robust Mean Absolute Deviation': rmad,
        'Root Mean Squared': rms,
        'Standard Deviation': std_dev,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Variance': variance,
        'Uniformity': uniformity
    }




import numpy as np
from skimage.feature import greycomatrix, greycoprops

def calculate_glcm_features(image):
    # Calculate co-occurrence matrix
    if isinstance(image, np.ndarray):
        glcm = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)

        # Calculate statistics
        contrast = greycoprops(glcm, 'contrast')
        correlation = greycoprops(glcm, 'correlation')
        energy = greycoprops(glcm, 'energy')
        homogeneity = greycoprops(glcm, 'homogeneity')
        asm = greycoprops(glcm, 'ASM')
        dissimilarity = greycoprops(glcm, 'dissimilarity')
    elif isinstance(image, torch.Tensor):
        glcm = calculate_glcm(image)
        contrast = calculate_contrast(glcm)
        correlation = calculate_correlation(glcm)
        energy = calculate_energy(glcm)
        homogeneity = calculate_homogeneity(glcm)
        asm = calculate_asm(glcm)
        dissimilarity = calculate_dissimilarity(glcm)

    return {
        'Contrast': contrast,
        'Correlation': correlation,
        'Energy': energy,
        'Homogeneity': homogeneity,
        'ASM': asm,
        'Dissimilarity': dissimilarity
    }

def calculate_glcm(image, distance=1, direction=0):
    # This is a simple and not very efficient implementation of GLCM
    image = torch.tensor(image, dtype=torch.int)
    glcm = torch.zeros((256, 256))
    for i in range(image.shape[0] - distance):
        for j in range(image.shape[1] - distance):
            glcm[image[i, j], image[i + distance, j + direction]] += 1
    glcm /= glcm.sum()
    return glcm

def calculate_contrast(glcm):
    contrast = 0
    for i in range(256):
        for j in range(256):
            contrast += (i - j) ** 2 * glcm[i, j]
    return contrast

def calculate_correlation(glcm):
    px = glcm.sum(dim=0)
    py = glcm.sum(dim=1)
    mean_x = torch.dot(px, torch.arange(256))
    mean_y = torch.dot(py, torch.arange(256))
    std_x = torch.sqrt(torch.dot(px, (torch.arange(256) - mean_x) ** 2))
    std_y = torch.sqrt(torch.dot(py, (torch.arange(256) - mean_y) ** 2))
    correlation = 0
    for i in range(256):
        for j in range(256):
            correlation += (i - mean_x) * (j - mean_y) * glcm[i, j]
    correlation /= std_x * std_y
    return correlation

def calculate_energy(glcm):
    return torch.sum(glcm ** 2)

def calculate_homogeneity(glcm):
    homogeneity = 0
    for i in range(256):
        for j in range(256):
            homogeneity += glcm[i, j] / (1 + abs(i - j))
    return homogeneity

def calculate_asm(glcm):
    return torch.sum(glcm ** 2)

def calculate_dissimilarity(glcm):
    dissimilarity = 0
    for i in range(256):
        for j in range(256):
            dissimilarity += abs(i - j) * glcm[i, j]
    return dissimilarity

def calculate_statistics_torch(image):
    glcm = calculate_glcm(image)
    return {
        'Contrast': calculate_contrast(glcm),
        'Correlation': calculate_correlation(glcm),
        'Energy': calculate_energy(glcm),
        'Homogeneity': calculate_homogeneity(glcm),
        'ASM': calculate_asm(glcm),
        'Dissimilarity': calculate_dissimilarity(glcm)
    }

def calculate_glrlm(image):
    # This is a simple and not very efficient implementation of GLRLM
    image = torch.tensor(image, dtype=torch.int)
    max_run_length = image.shape[1]
    glrlm = torch.zeros((256, max_run_length))
    for i in range(image.shape[0]):
        run_length = 0
        run_value = image[i, 0]
        for j in range(image.shape[1]):
            if image[i, j] == run_value:
                run_length += 1
            else:
                glrlm[run_value, run_length - 1] += 1
                run_length = 1
                run_value = image[i, j]
        glrlm[run_value, run_length - 1] += 1
    glrlm /= glrlm.sum()
    return glrlm

def calculate_short_run_emphasis(glrlm):
    sre = torch.sum(glrlm / (torch.arange(1, glrlm.shape[1] + 1) ** 2))
    return sre

def calculate_long_run_emphasis(glrlm):
    lre = torch.sum(glrlm * (torch.arange(1, glrlm.shape[1] + 1) ** 2))
    return lre

def calculate_low_gray_level_run_emphasis(glrlm):
    lglre = torch.sum(glrlm / (torch.arange(1, glrlm.shape[0] + 1) ** 2))
    return lglre

def calculate_high_gray_level_run_emphasis(glrlm):
    hglre = torch.sum(glrlm * (torch.arange(1, glrlm.shape[0] + 1) ** 2))
    return hglre

def calculate_statistics_torch(image):
    glrlm = calculate_glrlm(image)
    return {
        'Short Run Emphasis': calculate_short_run_emphasis(glrlm),
        'Long Run Emphasis': calculate_long_run_emphasis(glrlm),
        'Low Gray-Level Run Emphasis': calculate_low_gray_level_run_emphasis(glrlm),
        'High Gray-Level Run Emphasis': calculate_high_gray_level_run_emphasis(glrlm)
    }
    
def calculate_statistics(x,version):
    if version==1:
        o=torch.tensor([])
        for i in range(x.shape[0]): #for each batch
            channelfeatures=torch.tensor([])
            for j in range(x.shape[1]):
                jo=calculate_fos_features(x[i,j])
                dicttolist=lambda d: [d[key] for key in d]
                jo=torch.Tensor(dicttolist(jo))
                jo/=torch.max(jo)
                channelfeatures = torch.cat((channelfeatures,jo))
            if i==0:
                o=channelfeatures
            else:
                o=torch.vstack((o,channelfeatures))
        return o
