import numpy as np
import pytest
from jax_fht import fht, ifht
from jax import jacobian


def f(r, mu):
    return r ** (mu + 1) * np.exp(-(r ** 2) / 2)


def g(k, mu):
    return k ** (mu + 1) * np.exp(-(k ** 2) / 2)


@pytest.mark.parametrize("mu", [0, 1, 2])
def test_get_g(mu):
    logrmin = -4
    logrmax = 4
    logrc = (logrmin + logrmax) / 2
    n = 128
    nc = (n + 1) / 2.0
    dlogr = (logrmax - logrmin) / n
    dlnr = dlogr * np.log(10.0)
    r = 10 ** (logrc + (np.arange(1, n + 1) - nc) * dlogr)
    ar = f(r, mu)
    kr = 1
    logkc = np.log10(kr) - logrc
    k = 10 ** (logkc + (np.arange(1, n + 1) - nc) * dlogr)
    mask = g(k, mu) > 1.0e-4

    ak = fht(ar.copy(), dlnr, mu=mu)
    np.testing.assert_allclose(ak[mask], g(k, mu)[mask], atol=0.0001)


@pytest.mark.parametrize("mu", [0, 1, 2])
def test_get_f(mu):
    logkmin = -4
    logkmax = 4
    logkc = (logkmin + logkmax) / 2
    n = 128
    nc = (n + 1) / 2.0
    dlogk = (logkmax - logkmin) / n
    dlnk = dlogk * np.log(10.0)
    k = 10 ** (logkc + (np.arange(1, n + 1) - nc) * dlogk)
    ak = g(k, mu)
    rr = 1
    logrc = np.log10(rr) - logkc
    r = 10 ** (logrc + (np.arange(1, n + 1) - nc) * dlogk)
    mask = f(r, mu) > 1.0e-4

    ar = ifht(ak.copy(), dlnk, mu=mu)
    np.testing.assert_allclose(ar[mask], f(r, mu)[mask], atol=0.0001)


def test_get_g_derivative(mu=0):
    logrmin = -4
    logrmax = 4
    logrc = (logrmin + logrmax) / 2
    n = 128
    nc = (n + 1) / 2.0
    dlogr = (logrmax - logrmin) / n
    dlnr = dlogr * np.log(10.0)
    r = 10 ** (logrc + (np.arange(1, n + 1) - nc) * dlogr)
    ar = f(r, mu)
    jacobian(fht)(ar.copy(), dlnr, mu=mu)
