# Vectfit for python

## Introduction
This is `vectfit` implementation for Python, expanded and modified for python 3 from the [original Python 2 package by Phil Reynolds](https://github.com/PhilReinhold/vectfit_python). It is a duplication of the [vector fitting algorithm](https://www.sintef.no/projectweb/vectorfitting/) for MATLAB. The purpose of this module is to fit rational transfer functions that can be represented in the form

<img src="https://latex.codecogs.com/svg.latex?T(s)&space;=&space;\sum\limits_{m=1}^{N_p}&space;\frac{c_m}{s-a_m}&space;&plus;&space;d&space;&plus;&space;e&space;s," title="T(s) = \sum\limits_{m=1}^{N_p} \frac{c_m}{s-a_m} + d + e s," />

where $a_m$ are the poles, $c_m$ are the residues, while parameters $d$ ("offset") and $e$ ("differentiator" or "slope") are optional.

All credit goes to Bjorn Gustavsen for his MATLAB implementation, and the following papers:


 [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
     domain responses by Vector Fitting", IEEE Trans. Power Delivery,
     vol. 14, no. 3, pp. 1052-1061, July 1999.

 [2] B. Gustavsen, "Improving the pole relocating properties of vector
     fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
     July 2006.

 [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
     "Macromodeling of Multiport Systems Using a Fast Implementation of
     the Vector Fitting Method", IEEE Microwave and Wireless Components
     Letters, vol. 18, no. 6, pp. 383-385, June 2008.
     
## How to use
Unlike the original MATLAB package, the module is designed with aim of making the interaction with it as automatic and high-level as possible. The main interface for fitting is provided by a function `vectfit_auto()`. This functions runs the fitting process iteratively for a given number of iterations. It can either generate automatically a linear or logarithmic distribution of initial poles, or it can take an explicit distribution of initial poles as an argument. See `help(vectfit.vectfit_auto())` for details and an example of use below.

## Arguments for `vectfit_auto()`
| Argument | Default value | Meaning |
| --- | --- | --- |
| `n_complex_pairs` | 10 | Number of complex pole pairs to use |
| `n_real_poles` | 0 | Number of real poles to use |
| `n_complex_poles` | 10 | Number of complex poles to use (if `allow_nonconj==True`)|
| `init_spacing` | 'lin' | Initial spacing between complex poles or pole pairs ('lin' or 'log') |
| `loss_ratio` | 1e-2 | Loss ratio for the initial set of complex poles or pole pairs |
| `init_poles` | `[]` | Use user-provided initial poles (automatic generation is switched off) |
| `asymp` | 2 | The value controls the model as follows:<br> `1` -> $d=0$, $e=0$;<br>`2` -> $d\ne 0$, $e=0$;<br>`3` -> $d\ne 0$, $e\ne 0$|
| `allow_nonconj` | `False` | Allow isolated complex poles |
| `allow_unstable` | `False` | Allow unstable poles |
| `allow_rescale` | `True` | Scale the values down automatically if the matrix is ill-conditioned |
| `show` | `False` | Draw Bode plots of the original transfer function and the fit result |
| `track_poles` | `False` | Return the 2D list of poles, with rows representing pole sets at each iteration|
| `suppress_warnings` | `False` | Suppress `vectfit` warnings |



## Utilities
Other functions provided by the module:
* `model_polres(s, poles, residues, d=0, h=0)`: Evaluate a transfer function, defined by poles, residues, offset `d`, and slope `h`, at points given by `s`.
*  `model_zpk(s, zeros, poles, k)`: Evaluate a transfer function, defined by zeros, poles, and gain `k`, at points given by `s`.
* `vectfit.polres_to_zpk(poles, residues, d=0, h=0)`: Convert poles-residues representation into zeros, poles, and gain (zpk) representation. See section "From residues to zeros, poles, and gain (zpk)" below for details.
* `print_params(poles, residues, d=0, h=0, switch_to_Hz=False)`: Print the results, automatically convert to Hz if the corresponding evaluates to `True`.
* `print_zpk_params(zeros, poles, k, switch_to_Hz=False)`: Print the results, automatically convert to Hz if the corresponding evaluates to `True`.

## Fitting complex poles that do not form conjugated pairs
Fitting transfer functions with isolated complex pairs is enabled if `allow_nonconj=True` argument is specified. In this case, the number of generated initial complex poles is set by `n_complex_poles` instead of `n_complex_pairs`. See [example_cvectfit.ipynb](./example_cvectfit.ipynb) for details and a use example.

## Example of use
```python
import numpy as np
import matplotlib.pyplot as plt
import vectfit

# List of frequencies in Hz
freqs=np.logspace(-3,3,1000)
# List of arguments of the transfer function in the Laplace domain
s = 2*np.pi*freqs*1j

# Test transfer function 1, given by poles, residues, offset, and slope:
tst_poles = 2*np.pi*np.array([-1e-2+1e-2j,-1e-2-1e-2j,-1e-2+1j,-1e-2-1j,-1e1,-5e1]) # poles
tst_residues = -tst_poles # residues
tst_d=0 # offset (optional)
tst_h=0 # slope (optional)

# Evaluation of test function 1: list of complex values to fit
tst_tf = vectfit.model_polres(s, tst_poles, tst_residues, tst_d, tst_h)

# Vector fitting
# 1. using default parameters (10 iterations, 10 complex pole pairs), make a plot
fit1_poles, fit1_residues, fit1_d, fit1_h = \
    vectfit.vectfit_auto(tst_tf, s, show=True) 

# 2. giving an explicit number of poles (2 complex conjugated pairs and 2 real poles, 10 iterations)
fit2_poles, fit2_residues, fit2_d, fit2_h = \
    vectfit.vectfit_auto(tst_tf, s, n_complex_pairs=2, n_real_poles=2) 

print("Test parameters:")
print("================")
vectfit.print_params(tst_poles, tst_residues, tst_d, tst_h, switch_to_Hz = True)
print()

print("Fitted parameters:")
print("==================")
vectfit.print_params(fit2_poles, fit2_residues, fit2_d, fit2_h, switch_to_Hz = True)
```
## From residues to zeros, poles, and gain (zpk)

As in the original MATLAB package, results produced by `vectfit_auto()` are returned in the form of $N_p$ poles $a_m$, $N_p$ residues $c_m$, offset $d$, and slope $e$. This package, however, also includes function `polres_to_zpk()` that allows for switching into zpk formalism, so that the resulting transfer function is expressed as

<img src="https://latex.codecogs.com/svg.latex?T(s)&space;=&space;\sum\limits_{m=1}^{N_p}&space;\frac{c_m}{s-a_m}&space;&plus;&space;d&space;&plus;&space;e&space;s&space;=&space;k&space;\frac{\prod_{n=1}^{N_z}&space;(s-z_n)}{&space;\prod_{m=1}^{N_p}&space;(s-a_m)}," title="T(s) = \sum\limits_{m=1}^{N_p} \frac{c_m}{s-a_m} + d + e s = k \frac{\prod_{n=1}^{N_z} (s-z_n)}{ \prod_{m=1}^{N_p} (s-a_m)}," />

where $z_n$ are the zeros, $k$ is the total gain (that can be complex), and $N_z$ is the number of zeros. Note that due to the nature of the pole-residue representation, the number of zeros in the fit is fixed by the number of poles:
* If $d = 0$ and $e = 0$ then $N_z = N_p - 1$;
* If $d \ne 0$ and $e = 0$ then $N_z = N_p$;
* If $e \ne 0$ then $N_z = N_p + 1$

##### Special case of complex total gain k ($\mathrm{Im~} k \ne 0$)
A special case where `vectfit` does not work too well yet is the case where the transfer function has _complex_ total gain $k$, $\Im k \ne 0$. Due to internal representations of functions in matrix calculations, fitting in this case requires calling `vectfit_auto()` with `allow_unstable=True` argument setting, even if the resulting poles are stable, plus the number of specified complex pairs should be significantly higher than the number of actual complex pairs. This issue will be addressed in the future.

##### Evaluating zpk transfer functions
A transfer function that is defined by zeros, poles, and gain can be evaluated at points given by array `s` using function `model_zpk(s, zeros, poles, k)`.

## Example with zpk fitting
```python
import numpy as np
import matplotlib.pyplot as plt
import vectfit

# Defining transfer function with poles and zeros
tst_poles = 2*np.pi*np.array([-5, -1, -5-100j, -5+100j])
tst_zeros = 2*np.pi*np.array([-0.2, 4])
tst_k = 42.42 # total gain
# tst_k = 42.42+50j # complex total gain

# Evaluation of the test transfer function
tst_tf = vectfit.model_zpk(s, tst_zeros, tst_poles, tst_k)

# Fitting the transfer function
fit_poles, fit_residues, fit_d, fit_h = \
    vectfit.vectfit_auto(tst_tf, s, n_complex_pairs=1, n_real_poles=2, n_iter = 10, show=True) 

# # Replace the above with the commented function call below to fit the case of complex total gain k
# fit_poles, fit_residues, fit_d, fit_h = \
#     vectfit.vectfit_auto(tst_tf, s, n_complex_pairs=10, n_real_poles=2, n_iter = 100, allow_unstable=True, show=True) 

# Converting results into zpk
fit_zeros,fit_poles,fit_k = vectfit.polres_to_zpk(fit_poles, fit_residues, d=0, h=0)

print("Test parameters:")
print("================")
vectfit.print_zpk_params(tst_zeros, tst_poles, tst_k, switch_to_Hz = True)
print()

print("Fitted parameters:")
print("==================")
vectfit.print_zpk_params(fit_zeros, fit_poles, tst_k, switch_to_Hz = True)
```
# Vectfit modification for non-conjugated complex poles

## Introduction
If the signals that are present in a system are *complex*, then the impulse responses in this system can also be complex. Correspondingly, the transfer function of this system can include complex poles that do not form complex conjugated pairs (i.e., frequency response becomes asymmetric with respect to zero frequency). For example, optical systems, such as laser interferometers, are often described as operators acting on complex amplitudes of the electromagnetic field. Many results of control theory remain applicable in the complex domain without any changes. However, some calculation methods need to be modified in order to work in the complex domain.

Unlike the original MATLAB package, `vectfit` for python can fit system transfer functions for complex-valued signals. Mathematically, the difference between such transfer functions and those that describe systems with real signals, is that the latter can only have complex poles that form conjugated pairs, whilst the former can have lone complex poles, or complex pole "pairs" of the form <img src="https://latex.codecogs.com/svg.latex?s_p&space;=&space;\sigma_1&plus;i&space;(\omega_0\pm\omega_1)" title="s_p = \sigma_1+i (\omega_0\pm\omega_1)" /> (i.e. are "symmetric" with respect to some axis $\Im s = \omega_0$ instead of $\Im s = 0$). 

This generalisation is based on the idea for `vectfit` modification described the following paper:
 * Spina, D., Ye, Y., Deschrijver, D., Bogaerts, W. and Dhaene, T. (2021), Complex vector fitting toolbox: a software package for the modelling and simulation of general linear and passive baseband systems. Electron. Lett., 57: 404-406.

In that paper, the authors remove the conjugateness constrain on from the original `vectfit` approach, described in
 * Gustavsen, B. (2009). Fast passivity enforcement for S-parameter models by perturbation of residue matrix eigenvalues. IEEE Transactions on advanced packaging, 33(1), 257-265
     
## How to use
The functionality for complex-valued systems is included in the main tool of the module -- `vectfit_auto()` function. Passing the argument `allow_nonconj=True` to this function enables the complex vector fitting algorithm. In this case, an initial distribution of single complex poles is generated instead of complex conjugate pairs. The iterative process uses functions in which the requirement for the poles being in complex conjugated pairs is removed according to the method suggested in `Spina, D. et al, 2021` (referenced above).

## Example
```python
# List of frequencies in Hz
freqs=np.logspace(-3,3,1000)
# List of arguments of the transfer function in the Laplace domain
s = 2*np.pi*freqs*1j

# Test transfer function 1, given by poles, residues, offset, and slope:
tst_poles = np.array([-1e-2+1e-2j,-1e-2-1e-2j,-1e-2+1j,-1e1,-5e1])
tst_residues = -tst_poles
tst_d=0
tst_h=0

# Evaluation of test function: list of complex values to fit
tst_tf = vectfit.model_polres(s, tst_poles, tst_residues, tst_d, tst_h)

# Vector fitting
# 1. standard vector fitting (bad result even for many parameters and lots of iterations
# because we have a lone complex pole)
fit1_poles, fit1_residues, fit1_d, fit1_h = \
    vectfit.vectfit_auto(tst_tf, s, n_complex_pairs=20, n_real_poles=2, n_iter=100) 
# 2. using allow_nonconj (gives a reasonably good fit with small number of initial poles)
fit2_poles, fit2_residues, fit2_d, fit2_h = \
    vectfit.vectfit_auto(tst_tf, s, \
                         allow_nonconj=True, \
                         n_complex_poles=3, \
                         n_real_poles=2 \
                        ) 

fit1_tf = vectfit.model_polres(s, fit1_poles, fit1_residues, fit1_d, fit1_h)
fit2_tf = vectfit.model_polres(s, fit2_poles, fit2_residues, fit2_d, fit2_h)

# Plotting results
plt.figure(figsize=(12,8))
plt.subplot(211)
plt.title("Vectfitting a function with an isolated complex pole")
plt.loglog(freqs, np.abs(tst_tf), label = "original function", color='black')
plt.loglog(freqs, np.abs(fit1_tf), label = "standard vectfit", ls=":", color='b')
plt.loglog(freqs, np.abs(fit2_tf), label = "modified vectfit (allow_nonconj=True)", ls="--", color='r')
plt.legend()
plt.ylabel("Amplitude")
plt.subplot(212)
plt.semilogx(freqs, np.angle(tst_tf)*180/np.pi, label = "original function", color='black')
plt.semilogx(freqs, np.angle(fit1_tf)*180/np.pi, label = "standard vectfit", ls=":", color='b')
plt.semilogx(freqs, np.angle(fit2_tf)*180/np.pi, label = "modified vectfit (allow_nonconj=True)", ls="--", color='r')
plt.legend()
plt.xlabel("Frequency, Hz")
plt.ylabel("Phase, degrees")
plt.show()
```