import jax.numpy as jnp
from jax_fht import fht, ifht

# Based on https://github.com/DarkQuestCosmology/dark_emulator_public/blob/af71ff875d6a05462822c84cbc2a3791a8f5294b/dark_emulator/pyfftlog_interface/pyfftlog_class.py


class FFTLog:
    def __init__(
        self, num: int = 1, log_r_min: float = -3.0, log_r_max: float = 3.0, kr: int = 1
    ):
        """
        Class used to transform power spectra and correlation functions

        Parameters:
        -----------
        num (int): number of bins in fftlog in units of 2048
        log_r_min (float): Minimum r over which fftlog will be computed.
        log_r_max (float): Maximum r over which fftlog will be computed.
        kr (float): Produce of k_c and r_c where 'c' indicates the center of 
                    scale array, r and k.
        """
        self.n = 2048 * num
        self.kr = kr
        nc = float(self.n + 1.0) / 2.0

        log_rc = (log_r_min + log_r_max) / 2.0
        dlogr = (log_r_max - log_r_min) / self.n
        self.dlnr = dlogr * jnp.log(10.0)
        self.r = 10 ** (log_rc + (jnp.arange(1, self.n + 1) - nc) * dlogr)

        log_kc = jnp.log10(self.kr) - log_rc
        log_k_min = jnp.log10(self.kr) - log_r_max
        log_k_max = jnp.log10(self.kr) - log_r_min
        dlogk = (log_k_max - log_k_min) / self.n
        self.dlnk = dlogk * jnp.log(10.0)
        self.k = 10 ** (log_kc + (jnp.arange(1, self.n + 1) - nc) * dlogk)

    def pk2xi(self, pk: jnp.array, idx_shift: float = 0.0) -> jnp.array:
        """converting P(k) to xi(r)
        Converting 3-dimensional power spectrum into 3-dimensional correlation 
        function, defined as

        Parameters:
        -----------
        pk: power spectrum evaluated at ```self.k```
        idx_shift: power index, to tune fftlog convergence.

        Returns
        -------
        xi:  two point correlation function evaluated at ```self.r```
        """
        beta = -1.5 - idx_shift
        prefactor = (2.0 * jnp.pi) ** (-1.5) * self.r ** beta
        Ak = self.k ** (1.5 - idx_shift) * pk
        return prefactor * ifht(Ak, self.dlnk, mu=0.5)

    def xi2pk(self, xi: jnp.array, idx_shift: float = 0.0) -> jnp.array:
        """converting xi(r) to P(k)

        Parameters:
        -----------
        xi: two point correlation function evaluated at ```self.r```
        idx_shift: power index, to tune fftlog convergence.

        Returns
        -------
        pk:  power spectrum evaluated at ```self.k```
        """
        beta = -1.5 - idx_shift
        prefactor = (2.0 * jnp.pi) ** (1.5) * self.k ** beta
        Ak = self.r ** (1.5 - idx_shift) * xi
        return prefactor * fht(Ak, self.dlnr, mu=0.5)
