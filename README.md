# Maximum Discrepancy Non-Negative Matrix Factoziation for Single Channel Source Separation

This GitHub repository contains code related to the paper "Maximum Discrepancy Generative Regularization and Non-Negative Matrix Factorization for Single Channel Source Separation" written by Martin Ludvigsen and Markus Grasmair. The paper proposes a new approach to training NMF for single channel source separation (SCSS), using ideas from recent work on adversarial regularization functions. The code in this repository implements the proposed approach and can be used to reproduce the results presented in the paper, as well as to explore variations of the method and apply it to other datasets.

The main goal and novelty is to represent true data well with a non-negative basis, as well as adversarial data poorly.

In other words, we want $U \approx WH$ and $\hat{U} \neq W\hat{H}$, where:
- $U$ is a $m \times N$ matrix containing the true data stored column-wise.
- $\hat{U}$ is a $m \times \hat{N}$ matrix containing the adversarial data stored column-wise.
- $H$ is a $d \times N$ matrix containing the true weights.
- $\hat{H}$ is a $d \times \hat{N}$ matrix containing the adversarial weights.

MDNMF is fitted by solving

$$ \min_{W \ge 0} \frac{1}{N} \lVert U - WH(U,W)\rVert_F^2 - \frac{\tau_A}{\hat{N}}  \lVert\hat{U} - WH(\hat{U},W)\rVert_F^2 + \gamma |W|_1, $$

where

$$ H(U,W) = \arg \min_{H \ge 0} \lVert U - WH\rVert_F^2 + \lambda |H|_1.$$

The parameter $\tau_A \ge 0$ is the so-called adversarial weight, representing how concerned we are with fitting true data versus not fitting adversarial data. Small values yield good fits for true data. In particular, $\tau_A = 0$ is just standard NMF. Large values yield worse fits for adversarial data.

The main application of this method is for single channel source separation problems, but can be applied to any inverse problem where the true signals can be reasonably represented with non-negative bases.

## Dependencies and datasets

The code is implemented in Python, and the dependencies are the packages NumPy, Librosa and Pandas. 

The datasets used in numerical experiments can be obtained as follows:
- The MNIST dataset is imported using the Python package Tensorflow/Keras. 
- The LibriSpeech dataset can be obtained here: https://www.openslr.org/12. We only use the dev-clean part of the dataset.
- The Musan dataset can be obtained here: https://www.openslr.org/17/. We only use the noise part of the dataset.

## Usage
The interface and usage is relatively similar to the scikit learn implementation of NMF and similar methods.

There are two main classes:

- ```NMF```, which is an object that handles fitting of NMF bases for a single source.
- ```NMF_separation```, which is an object that handles fitting and separation for a specific source separation problem using NMF.

For example, to fit a standard NMF with data stored column-wise in $U$

```python
d = 32 # Number of basis vectors

# 100 epochs with batch size 500 
nmf = NMF(d = d, batch_size = 500, epochs = 100)

# Standard fitting
nmf.fit_std(U)
```

We can then extract the basis/dictionary $W$ and the weights/latent variables $H$:
```python
W = nmf.W
H = nmf.H

U_reconstructed = np.dot(W, H)
```

We can alternatively recalculate $H$ via transformation:
```python
H = nmf.transform(U)
```

For adversarial fitting, we can do
```python
nmf = NMF(d = d, batch_size = 500, epochs = 100, prob = "adv", tau_A = 0.1)

nmf.fit_adv(U, U_hat)
```

## Results

Coming soon.

## Citation

Coming soon. 
