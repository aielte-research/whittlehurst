from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="whittlehurst",
    version="0.2",
    author="Bálint Csanády",
    author_email="csbalint@protonmail.ch",
    license="MIT",
    description="Hurst exponent estimation usding Whittle's method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aielte-research/whittlehurst.git",
    keywords="hurst fractal Gaussian noise Brownian motion ARFIMA econometrics time-series",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    install_requires=[
        "numpy",
        "scipy"
    ],
)