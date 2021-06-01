"""
vectfit implementation for Python, modified for python3 by Artemiy Dmitriev <artemiydmitriev@gmail.com>
from the original Python 2 package by Phil Reynolds (https://github.com/PhilReinhold/vectfit_python)

Duplication of the vector fitting algorithm in python (http://www.sintef.no/Projectweb/VECTFIT/)

All credit goes to Bjorn Gustavsen for his MATLAB implementation, and the following papers


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
"""
__author__ = 'Artemiy Dmitriev'

import numpy as np
from numpy.polynomial import polynomial as P
import warnings

class IllCondError(Exception):
    pass

def cc(z):
    return z.conjugate()

def model_polres(s, poles, residues, d=0, h=0):
    """Evaluates a transfer function given by poles, residues, offset `d`, and slope `h`, at all points from `s`"""
    my_sum = np.sum(residues/(np.tile(s,(len(poles),1)).transpose()-poles),axis=1)
    return my_sum + d + s*h

def model_zpk(s, zeros, poles, k):
    """Evaluates a transfer function given by zeros, poles, and gain `k`, at all points from `s` """
    num_poly = P.polyfromroots(zeros)
    den_poly = P.polyfromroots(poles)
    num = P.polyval(s, num_poly)
    den = P.polyval(s, den_poly)
    return k * num / den

def vectfit_step(f, s, poles, allow_unstable=False):
    """
    f = complex data to fit
    s = j*frequency
    poles = initial poles guess
        note: All complex poles must come in sequential complex conjugate pairs
    returns adjusted poles
    """
    N = len(poles)
    Ns = len(s)

    cindex = np.zeros(N)
    # cindex is:
    #   - 0 for real poles
    #   - 1 for the first of a complex-conjugate pair
    #   - 2 for the second of a cc pair
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 3: #(i-1)th pole is real or has already been paired with (i-2)th pole
                if i+1 < N: #current pole is not the last one, deferring the check
                    cindex[i]=3
                else: #current pole is the last one
                    raise RuntimeError("Complex poles must come in conjugate pairs: %s" % (poles[i]))
            else: # (i-1)th pole is complex and has not formed a pair yet
                if cc(poles[i-1]) == p:
                    # we identified both poles from a complex conjugate pair
                    cindex[i-1]=1
                    cindex[i]=2
                else:
                    raise RuntimeError("Complex poles must come in conjugate pairs: %s, %s" % (poles[i-1],poles[i]))
                    
    # First linear equation to solve. See Appendix A
    A = np.zeros((Ns, 2*N+2), dtype=np.complex128)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

        A [:, N+2+i] = -A[:, i] * f

    A[:, N] = 1
    A[:, N+1] = s

    # Solve Ax == b using pseudo-inverse
    b = f
    A = np.vstack((np.real(A), np.imag(A)))
    b = np.concatenate((np.real(b), np.imag(b)))
    x, residuals, rnk, s = np.linalg.lstsq(A, b, rcond=-1)

    residues = x[:N]
    d = x[N]
    h = x[N+1]

    # We only want the "tilde" part in (A.4)
    x = x[-N:]

    # Calculation of zeros: Appendix B
    A = np.diag(poles)
    b = np.ones(N)
    c = x
    for i, (ci, p) in enumerate(zip(cindex, poles)):
        if ci == 1:
            x, y = np.real(p), np.imag(p)
            A[i, i] = A[i+1, i+1] = x
            A[i, i+1] = -y
            A[i+1, i] = y
            b[i] = 2
            b[i+1] = 0

    H = A - np.outer(b, c)
    H = np.real(H)
    new_poles = np.sort(np.linalg.eigvals(H))
    
    if allow_unstable == False: # default behaviour
        # flip the unstable poles, if there are any
        unstable = np.real(new_poles) > 0
        new_poles[unstable] -= 2*np.real(new_poles)[unstable]
    return new_poles

def calculate_residues(f, s, poles, rcond=-1, allow_rescale=False):
    Ns = len(s)
    N = len(poles)

#     cindex = np.zeros(N)
#     for i, p in enumerate(poles):
#         if p.imag != 0:
#             if i == 0 or cindex[i-1] != 1:
#                 assert cc(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % poles[i:i+1])
#                 cindex[i] = 1
#             else:
#                 cindex[i] = 2

    cindex = np.zeros(N)
    # cindex is:
    #   - 0 for real poles
    #   - 1 for the first of a complex-conjugate pair
    #   - 2 for the second of a cc pair
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 3: #(i-1)th pole is real or has already been paired with (i-2)th pole
                if i+1 < N: #current pole is not the last one, deferring the check
                    cindex[i]=3
                else: #current pole is the last one
                    raise RuntimeError("Complex poles must come in conjugate pairs: %s" % (poles[i]))
            else: # (i-1)th pole is complex and has not formed a pair yet
                if cc(poles[i-1]) == p:
                    # we identified both poles from a complex conjugate pair
                    cindex[i-1]=1
                    cindex[i]=2
                else:
                    raise RuntimeError("Complex poles must come in conjugate pairs: %s, %s" % (poles[i-1],poles[i]))

    # use the new poles to extract the residues
    A = np.zeros((Ns, N+2), dtype=np.complex128)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

    A[:, N] = 1
    A[:, N+1] = s
    
    # Solve Ax == b using pseudo-inverse
    b = f
    A = np.vstack((np.real(A), np.imag(A)))
    b = np.concatenate((np.real(b), np.imag(b)))
    cA = np.linalg.cond(A)
    if cA > 1e13:
        if allow_rescale==True:
            raise IllCondError(cA)
        else:
            warnings.warn("Ill Conditioned Matrix. Consider scaling the problem down \nCond(A)={}".format(cA))
    x, residuals, rnk, s = np.linalg.lstsq(A, b, rcond=rcond)

    # Recover complex values
    x = np.complex128(x)
    for i, ci in enumerate(cindex):
       if ci == 1:
           if cindex[i+1] != 2:
                raise RuntimeError("cindex[%s] = %s, cindex[%s] = %s" % (i, ci,i+1, cindex[i+1]))
           r1, r2 = x[i:i+2]
           x[i] = r1 - 1j*r2
           x[i+1] = r1 + 1j*r2

    residues = x[:N]
    d = x[N].real
    h = x[N+1].real
    return residues, d, h

def print_params(poles, residues, d=0, h=0, switch_to_Hz=False):
    """Print poles, residues, offset, and slope"""
    residues = np.asarray(residues)
    poles = np.asarray(poles)
    if switch_to_Hz == True:
        norm = 2*np.pi
        with np.printoptions(precision=4, suppress=True):
            print("poles [Hz]   :\n", poles/norm)
            print("residues [Hz]:\n", residues/norm)
            print("offset       :\n", d)
            print("slope [1/Hz] :\n", h*norm)
    else:
        with np.printoptions(precision=4, suppress=True):
            print("poles [rad/s]   :\n", poles)
            print("residues [rad/s]:\n", residues)
            print("offset          :\n", d)
            print("slope [s/rad]   :\n", h)

def print_zpk_params(zeros, poles, k, switch_to_Hz=False):
    """Print zpk"""
    zeros = np.asarray(zeros)
    poles = np.asarray(poles)
    if switch_to_Hz == True:
        norm = 2*np.pi
        with np.printoptions(precision=4, suppress=True):
            print("zeros [Hz]   :\n", zeros/norm)
            print("poles [Hz]   :\n", poles/norm)
            print("k            :\n", k)
    else:
        with np.printoptions(precision=4, suppress=True):
            print("zeros [rad/s]   :\n", zeros/norm)
            print("poles [rad/s]   :\n", poles/norm)
            print("k               :\n", k)

def vectfit_auto(f, s, n_complex_pairs=10, n_real_poles=0, n_iter=10, show=False,
                  init_spacing="lin", loss_ratio=1e-2, rcond=-1, track_poles=False, allow_unstable=False, allow_rescale=True):
    w = np.imag(s)
    if init_spacing=="lin":
        pole_locs = np.linspace(w[0], w[-1], n_complex_pairs+2)[1:-1]
    elif init_spacing=="log":
        pole_locs = np.geomspace(w[0], w[-1], n_complex_pairs+2)[1:-1]
    else:
        raise RuntimeError("Acceptable values for init_spacing are 'lin' and 'log'")

    lr = loss_ratio
    init_poles = poles = np.concatenate([[p*(-lr + 1j), p*(-lr - 1j)] for p in pole_locs])

    if n_real_poles != 0:
        poles = np.concatenate((poles, n_real_poles*[-1]))

    poles_list = []
    for _ in range(n_iter):
        poles = vectfit_step(f, s, poles, allow_unstable=allow_unstable)
        poles_list.append(poles)

    try:
        residues, d, h = calculate_residues(f, s, poles, rcond=rcond, allow_rescale=allow_rescale)
    except IllCondError as inst:
        cA, = inst.args
        warnings.warn("Ill-conditioned matrix, Cond(A)={}. Attempting automatic rescaling".format(cA))
        s_scale = np.abs(s[-1])
        f_scale = np.abs(f[-1])
        
        if track_poles:
            poles_s, residues_s, d_s, h_s, poles_list_s = vectfit_auto(f / f_scale, s / s_scale, n_complex_pairs=n_complex_pairs, n_real_poles=n_real_poles, n_iter=n_iter, show=False, init_spacing=init_spacing, loss_ratio=loss_ratio, rcond=rcond, track_poles=track_poles, allow_unstable=allow_unstable, allow_rescale=False)
            poles_list = poles_list_s * s_scale
        else:
            poles_s, residues_s, d_s, h_s = vectfit_auto(f / f_scale, s / s_scale, n_complex_pairs=n_complex_pairs, n_real_poles=n_real_poles, n_iter=n_iter, show=False, init_spacing=init_spacing, loss_ratio=loss_ratio, rcond=rcond, track_poles=track_poles, allow_unstable=allow_unstable, allow_rescale=False)
            
        poles = poles_s * s_scale
        residues = residues_s * f_scale * s_scale
        d = d_s * f_scale
        h = h_s * f_scale / s_scale
        
    if show == True:
        # Evaluating fitted transfer function
        fit = model_polres(s, poles, residues, d, h)
        
        # Plotting results
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,8))
        plt.subplot(211)
        plt.title("Vectfit result")
        plt.loglog(w/2/np.pi, np.abs(f), label = "original function")
        plt.loglog(w/2/np.pi, np.abs(fit), label = "automatic fit", ls="--" )
        plt.legend()
        plt.ylabel("Amplitude")
        plt.subplot(212)
        plt.semilogx(w/2/np.pi, np.angle(f)*180/np.pi, label = "original function")
        plt.semilogx(w/2/np.pi, np.angle(fit)*180/np.pi, label = "automatic fit", ls="--")
        plt.legend()
        plt.xlabel("Frequency, Hz")
        plt.ylabel("Phase, degrees")
        plt.show()
        
    if track_poles:
        return poles, residues, d, h, np.array(poles_list)
    
    return poles, residues, d, h

def polres_to_zpk(poles, residues, d=0, h=0):
    """Converts the result given by vectfit into zpk (zeros, poles, gain)
    Number of returned zeros for N given poles is:
    * N-1 if d  = 0 and h = 0,
    * N   if d != 0 and h = 0,
    * N+1 if h != 0."""
    
    poles = poles # poles are not changed
    N = len(poles) # number of poles
    
    zeros_poly = P.polyzero
    for p,r,i in zip(poles,residues,range(N)):
        this_iter_roots = np.delete(poles, i) # Array of roots, excluding the current pole
        this_iter_poly = r * P.polyfromroots(this_iter_roots) 
        zeros_poly = P.polyadd(zeros_poly, this_iter_poly)
        
    if d != 0:
        offset_poly = d * P.polyfromroots(poles)
        zeros_poly = P.polyadd(zeros_poly, offset_poly)
    if h != 0:
        slope_polyroots = np.append(poles, 0)
        slope_poly = h * P.polyfromroots(slope_polyroots)
        zeros_poly = P.polyadd(zeros_poly, slope_poly)
    
    # normalising zeros by the coefficient of highest-order term
    k = zeros_poly[-1]
    zeros_poly = zeros_poly / k
    
    zeros = P.polyroots(zeros_poly)
    
    return zeros,poles,k

if __name__ == '__main__':
    freqs=np.logspace(-3,3,1000)

    my_poles = np.array([-1e-2+1e-2j,-1e-2-1e-2j,-1e-2+1j,-1e-2-1j,-1e1,-5e1])
    my_residues = -my_poles
    my_d=0
    my_h=0

    print("Using the test parameter set:")
    print_params(my_poles, my_residues, my_d, my_h)

    my_tf = model_polres(1j*freqs, my_poles, my_residues, my_d, my_h) # Test TF

    print("Fitting...")
    uf_poles, uf_residues, uf_d, uf_h = vectfit_auto(my_tf,
                                                             1j*freqs,
                                                             init_spacing='log')
    fit_unscaled = model_polres(1j*freqs, uf_poles, uf_residues, uf_d, uf_h) # Unscaled vector fitting

    sf_poles, sf_residues, sf_d, sf_h = vectfit_auto_rescale(my_tf,
                                                                     1j*freqs,
                                                                     init_spacing='log')
    fit_scaled = model(1j*freqs, sf_poles, sf_residues, sf_d, sf_h)
    print("Complete")

    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.title("Example: fitting a transfer function with vectfit in Python")
    plt.loglog(freqs, np.abs(my_tf), label = "model")
    plt.loglog(freqs, np.abs(fit_unscaled), label = "fit_unscaled", ls="--" )
    plt.loglog(freqs, np.abs(fit_scaled), label = "fit_scaled", ls=":")
    plt.legend()
    plt.ylabel("Amplitude")
    plt.subplot(212)
    plt.semilogx(freqs, np.angle(my_tf)*180/np.pi, label = "model")
    plt.semilogx(freqs, np.angle(fit_unscaled)*180/np.pi, label = "fit_unscaled", ls="--")
    plt.semilogx(freqs, np.angle(fit_scaled)*180/np.pi, label = "fit_scaled", ls=":")
    plt.legend()
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Phase, degrees")
    plt.show()
