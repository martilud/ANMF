import matplotlib.pyplot as plt
import numpy as np

# Code for plotting images
def plotimgs(imgs, nplot = 8, rescale = False, filename = None):
    """
    Plots nplot*nplot images on an nplot x nplot grid. 
    Saves to given filename if filename is given
    Can also rescale the RGB channels
    input:
        imgs: (N,height,width) array containing images, where N > nplot**2
        nplot: integer, nplot**2 images will be plotted
        rescale: bool
        filename: string, figure will be saved to this location. Should end with ".png".
    """
    # We will change some of the parameters of matplotlib, so we store the initial ones
    oldparams = plt.rcParams['figure.figsize']

    # New params toi make better plot. There definitely exists better ways of doing this
    plt.rcParams['figure.figsize'] = (16, 16)

    # Initialize subplots
    fig, axes = plt.subplots(nplot,nplot)

    # Set background color
    plt.gcf().set_facecolor("lightgray")

    # Iterate over images
    for idx in range(nplot**2):
        
        # Indices
        i = idx//nplot; j = idx%nplot

        # Remove axis
        axes[i,j].axis('off')

        axes[i,j].imshow(imgs[idx,:,:], cmap = "gray")
    
    # Plot
    plt.tight_layout()

    # Save if filename is given
    if filename is not None:
        plt.savefig(filename)

    plt.show()

    # Return to old parameters
    plt.rcParams['figure.figsize'] = oldparams

def wiener(u, v, eps = 1e-7):
    """
    Input:
        u: (N,M,...), numpy array that is to be filtered
        v: (N,...), numpy array 
        eps: float, safe division parameter
    """
    out = np.zeros_like(u)
    for i in range(out.shape[0]):
        usum = np.sum(u[i], axis = 0)
        for j in range(out.shape[1]):
            out[i,j] = v[i] * u[i,j]/(usum + eps)
    return out

def PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2, axis=0)
    #if mse == 0:
    #    return 100
    PIXEL_MAX = np.max(img1, axis = 0)
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_snr(clean_signal, noisy_signal):
    """
    Calculates the Signal-to-Noise Ratio (SNR) between a clean and noisy signal.

    Args:
        clean_signal (np.ndarray): 1D array representing the clean signal
        noisy_signal (np.ndarray): 1D array representing the noisy signal

    Returns:
        float: The SNR in decibels (dB)
    """
    # Ensure that both signals are of the same length
    assert len(clean_signal) == len(noisy_signal), "Clean and noisy signals should have the same length."

    # Compute the power of the clean signal
    clean_power = np.sum(np.square(clean_signal))

    # Compute the power of the noise signal
    noise = clean_signal - noisy_signal
    noise_power = np.sum(np.square(noise))

    # Compute the SNR in dB
    snr_db = 10 * np.log10(clean_power / noise_power)

    return snr_db

def calculate_power(signal):
    return np.sum(np.square(signal))

def calculate_sdr(clean_signal, noisy_signal):

    assert len(clean_signal) == len(noisy_signal)

    scaling_factor = np.dot(noisy_signal, clean_signal) / np.linalg.norm(clean_signal)**2

    return 10 * np.log10( np.linalg.norm(scaling_factor * clean_signal)**2 /np.linalg.norm(scaling_factor * clean_signal - noisy_signal)**2  )

    