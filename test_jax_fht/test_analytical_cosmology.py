import jax.numpy as jnp
import pytest
from jax_fht.cosmology import FFTLog
from jax import grad, jacobian
import numpy as np


def xi(r, A=1.0):
    return A * jnp.exp(-(r ** 2))


def pk(k, A=1.0):
    return A * (jnp.pi ** 1.5) * jnp.exp(-(k ** 2) / 4.0)


@pytest.fixture
def fftlog():
    return FFTLog(num=1, log_r_min=-4.0, log_r_max=4.0)


def test_pk2xi(fftlog):
    xi_numerical = fftlog.pk2xi(pk(fftlog.k))
    r_mask = (fftlog.r > 1.0e-4) & (fftlog.r < 1)
    np.testing.assert_allclose(xi(fftlog.r)[r_mask], xi_numerical[r_mask], rtol=0.11)


def test_xi2pk(fftlog):
    pk_numerical = fftlog.xi2pk(xi(fftlog.r))
    k_mask = (fftlog.k > 1.0e-3) & (fftlog.k < 5)
    np.testing.assert_allclose(pk(fftlog.k)[k_mask], pk_numerical[k_mask], rtol=0.1)


@pytest.mark.parametrize("A", [2., 10., 20.])
def test_derivative(A, fftlog):
    k_mask = (fftlog.k > 1.0e-3) & (fftlog.k < 5)
    get_pk = lambda norm: fftlog.xi2pk(xi(fftlog.r, norm))

    np.testing.assert_allclose(
        jacobian(get_pk)(A)[k_mask], pk(fftlog.k[k_mask]), rtol=0.1
    )
