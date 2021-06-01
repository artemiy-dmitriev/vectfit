vectfit.py
==========

Duplication of the [Fast Relaxed Vector-Fitting algorithm](https://www.sintef.no/projectweb/vectorfitting/) in python.

Adapted for python3 from the original Python 2 package by Phil Reynolds (https://github.com/PhilReinhold/vectfit_python)

Only the minimal necessary changes were introduced into the original package. There is much room for code improvement.

To use, put vectfit.py somewhere on your path.

Use example (see also example.ipynb):
```python
# List of frequencies in Hz
freqs=np.logspace(-3,3,1000)
# List of arguments of the transfer function in the Laplace domain
s = 2*np.pi*freqs*1j

# Test transfer function 1, given by poles, residues, offset, and slope:
#given in zpk format (zeros, poles, gain):
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
