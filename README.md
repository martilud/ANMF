<!-- Include MathJax library -->
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# Adversarially Generated NMF for Single Channel Source Separation

This GitHub repository contains code related to the paper "Adversarially Generated NMF for Single Channel Source Separation" written by Martin Ludvigsen and Markus Grasmair. The paper proposes a new approach to training NMF for single channel source separation (SCSS), using ideas from recent work on adversarial regularization functions. The code in this repository implements the proposed approach and can be used to reproduce the results presented in the paper, as well as to explore variations of the method and apply it to other datasets.

The methods are implemented in Python using NumPy as backend. Thus, the methods should primarily be used on NumPy arrays.

## Dependencies and datasets

The dependencies are the Python packages NumPy, Librosa and Pandas. 

The datasets used in numerical experiments can be obtained as follows:
- The MNIST dataset is imported using the Python package Tensorflow/Keras. 
- The LibriSpeech dataset can be obtained here: https://www.openslr.org/12. We only use the dev-clean part of the dataset.
- The WHAM! dataset can be obtained here: https://wham.whisper.ai/.

## Non-negative Matrix factorization.

The main goal is to find non-negative $m \times d$ matrix $W$ that can represent true data $U$ well and adversarial data $\hat{U} poorly.

In other words, we want $U \approx WH$ and $\hat{U} \neq W\hat{H}$, where:
- $U$ is a $m \times N$ matrix containing the true data stored column-wise.
- $\hat{U}$ is a $m \times \hat{N}$ matrix containing the adversarial data stored column-wise.
- $H$ is a $d \times N$ matrix containing the true weights.
- $\hat{H}$ is a $d \times \hat{N}$ matrix containing the adversarial weights.

ANMF is fitted by solving

$$ \min_{W \ge 0} \frac{1}{N} \lVert U - WH(U,W)\rVert_F^2 - \frac{\tau_A}{\hat{N}}  \lVert\hat{U} - WH(\hat{U},W)\rVert_F^2, $$

where

$$ H(U,W) = \arg \min_{H \ge 0} \lVert U - WH\rVert_F^2.$$

The parameter $\tau_A$ is the so-called adversarial weight.

## Usage
The interface and usage is relatively similar to the scikit learn implementation of NMF and similar methods.

There are two main classes:

- ```NMF```, which is an object that handles fitting of NMF bases for a single source.
- ```NMF_separation```, which is an object that handles fitting and separation for a specific source separation problem using NMF.




For example, to fit a standard NMF with data stored column-wise in ```U```
```python
d = 32 # Number of basis vectors

# 100 epochs with batch size 500 where W is normalized between epochs
nmf = NMF(d = d, batch_size = 500, epochs = 100, normalize = True, mu_H = )

# Standard fitting
nmf.fit_std(U)
```

We can then extract the basis/dictionary $W$ and the weights/latent variables $H$:
```
W = nmf.W
H = nmf.H_r

U_reconstructed = np.dot(W, H)
```

We can alternatively recalculate H via transformation:
```
H = nmf.transform(U)
```


## Results

Coming soon.

## Citation

Coming soon. 
