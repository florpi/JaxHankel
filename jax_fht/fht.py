import jax.numpy as jnp
import numpy as np
from scipy.special import loggamma
from warnings import warn

# TODO: replace loggamma by jax loggamma

LN_2 = jnp.log(2)


def _fhtq(a, u, inverse=False):
    """Compute the biased fast Hankel transform.
    This is the basic FFTLog routine.
    """

    # size of transform
    n = jnp.shape(a)[-1]

    # check for singular transform or singular inverse transform
    if jnp.isinf(u[0]) and not inverse:
        warn(f"singular transform; consider changing the bias")
        # fix coefficient to obtain (potentially correct) transform anyway
        u = u.copy()
        u[0] = 0
    elif u[0] == 0 and inverse:
        warn(f"singular inverse transform; consider changing the bias")
        # fix coefficient to obtain (potentially correct) inverse anyway
        u = u.copy()
        u[0] = jnp.inf

    # biased fast Hankel transform via real FFT
    A = jnp.fft.rfft(a, axis=-1)
    if not inverse:
        # forward transform
        A *= u
    else:
        # backward transform
        A /= u.conj()
    A = jnp.fft.irfft(A, n, axis=-1)
    A = A[..., ::-1]

    return A


def fhtcoeff(n, dln, mu, offset=0.0, bias=0.0):
    """Compute the coefficient array for a fast Hankel transform.
    """

    lnkr, q = offset, bias

    # Hankel transform coefficients
    # u_m = (kr)^{-i 2m pi/(n dlnr)} U_mu(q + i 2m pi/(n dlnr))
    # with U_mu(x) = 2^x Gamma((mu+1+x)/2)/Gamma((mu+1-x)/2)
    xp = (mu + 1 + q) / 2
    xm = (mu + 1 - q) / 2
    y = jnp.linspace(0, jnp.pi * (n // 2) / (n * dln), n // 2 + 1)
    u = xm + y * 1j
    v = np.empty(n // 2 + 1, dtype=complex)
    loggamma(u, out=v)
    u = xp + y * 1j
    u = np.array(u)
    loggamma(u, out=u)
    u = jnp.array(u)
    y *= 2 * (LN_2 - lnkr)
    u = (u.real - v.real + LN_2 * q) + (u.imag + v.imag + y) * 1j
    u = jnp.exp(u)

    # fix last coefficient to be real
    u = u.real + u.imag.at[-1].set(0) * 1j

    # deal with special cases
    if not jnp.isfinite(u[0]):
        # write u_0 = 2^q Gamma(xp)/Gamma(xm) = 2^q poch(xm, xp-xm)
        # poch() handles special cases for negative integers correctly
        u[0] = 2 ** q * poch(xm, xp - xm)
        # the coefficient may be inf or 0, meaning the transform or the
        # inverse transform, respectively, is singular

    return u


def fht(a, dln, mu, offset=0.0, bias=0.0):
    n = jnp.shape(a)[-1]
    # bias input array
    if bias != 0:
        # a_q(r) = a(r) (r/r_c)^{-q}
        j_c = (n - 1) / 2
        j = jnp.arange(n)
        a = a * jnp.exp(-bias * (j - j_c) * dln)

    u = fhtcoeff(n, dln, mu, offset=offset, bias=bias)
    # transform
    A = _fhtq(a, u)

    # bias output array
    if bias != 0:
        # A(k) = A_q(k) (k/k_c)^{-q} (k_c r_c)^{-q}
        A *= jnp.exp(-bias * ((j - j_c) * dln + offset))

    return A


def ifht(A, dln, mu, offset=0.0, bias=0.0):
    r"""Compute the inverse fast Hankel transform.
    Computes the discrete inverse Hankel transform of a logarithmically spaced
    periodic sequence. This is the inverse operation to `fht`.
    Parameters
    ----------
    A : array_like (..., n)
        Real periodic input array, uniformly logarithmically spaced.  For
        multidimensional input, the transform is performed over the last axis.
    dln : float
        Uniform logarithmic spacing of the input array.
    mu : float
        Order of the Hankel transform, any positive or negative real number.
    offset : float, optional
        Offset of the uniform logarithmic spacing of the output array.
    bias : float, optional
        Exponent of power law bias, any positive or negative real number.
    Returns
    -------
    a : array_like (..., n)
        The transformed output array, which is real, periodic, uniformly
        logarithmically spaced, and of the same shape as the input array.
    See Also
    --------
    fht : Definition of the fast Hankel transform.
    fhtoffset : Return an optimal offset for `ifht`.
    Notes
    -----
    This function computes a discrete version of the Hankel transform
    .. math::
        a(r) = \int_{0}^{\infty} \! A(k) \, J_\mu(kr) \, r \, dk \;,
    where :math:`J_\mu` is the Bessel function of order :math:`\mu`.  The index
    :math:`\mu` may be any real number, positive or negative.
    See `fht` for further details.
    """

    # size of transform
    n = jnp.shape(A)[-1]

    # bias input array
    if bias != 0:
        # A_q(k) = A(k) (k/k_c)^{q} (k_c r_c)^{q}
        j_c = (n - 1) / 2
        j = jnp.arange(n)
        A = A * jnp.exp(bias * ((j - j_c) * dln + offset))

    # compute FHT coefficients
    u = fhtcoeff(n, dln, mu, offset=offset, bias=bias)

    # transform
    a = _fhtq(A, u, inverse=True)

    # bias output array
    if bias != 0:
        # a(r) = a_q(r) (r/r_c)^{q}
        a /= jnp.exp(-bias * (j - j_c) * dln)

    return a
