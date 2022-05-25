# JaxHankel
A Hankel transform implementation in jax, based in scipy's implementation

## Examples
In cosmoogy, use to convert power spectrum into correlation functions and vice-versa,

```python
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax_fht.cosmology import FFTLog


def xi(r, A=1.0):
    return A * jnp.exp(-(r ** 2))
    
fftlog = FFTLog(num=1, log_r_min=-4.0, log_r_max=4.0)
pk = fftlog.xi2pk(xi(fftlog.r))

plt.loglog(fftlog.k, pk)
```
Note that it is vectorized along the last dimension.

Thanks to jax we can now compute derivatives too, see for instance the derivative of the power spectrum respect to its norm (A),


```python

get_pk = lambda norm: fftlog.xi2pk(xi(fftlog.r, norm))
derivative = jacobian(get_pk)(5.)

```

## Install

```bash
$ pip install -e .
```

