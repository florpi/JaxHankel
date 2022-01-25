from distutils.core import setup

setup(
    name='JaxHankel',
    author="Carolina Cuesta-Lazaro",
    author_email="carolina.cuesta-lazaro@durham.ac.uk",
    version='0.1dev',
    url='https://github.com/florpi/JaxHankel',
    packages=['jax_fht'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),

    install_requires=[
                    'numpy',
                    'jupyter',
                    'h5py',
                    'pandas',
                    'scipy',
                    'seaborn',
                    'matplotlib',
                    'halotools',
                    ],

    zip_safe=False
)

