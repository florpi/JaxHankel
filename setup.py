from setuptools import setup
from os.path import abspath, dirname, join

this_dir = abspath(dirname(__file__))
with open(join(this_dir, "requirements.txt")) as f:
    requirements = f.read().split("\n")

setup(
    name='JaxHankel',
    author="Carolina Cuesta-Lazaro",
    author_email="carolina.cuesta-lazaro@durham.ac.uk",
    version='0.1',
    url='https://github.com/florpi/JaxHankel',
    packages=['jax_fht'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    zip_safe=False
)

