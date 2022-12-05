from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

with open('README.md', mode='r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='alphaminer',
    version='0.0.1',
    description='Trading Alphas Digging and Backtesting',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='OpenDILab Contributors',
    author_email='opendilab@pjlab.org.cn',
    url='https://github.com/opendilab/DI-engine',
    license='Apache License, Version 2.0',
    keywords='Trading Alphas Backtesting',
    packages=[
        # framework
        *find_packages(include=('alphaminer', "alphaminer.*")),
    ],
    python_requires=">=3.7",
    install_requires=[
        'numpy>=1.12.0',
        'pandas>=0.25.1',
        'joblib>=1.1.0',
        'easydict',
        'graphviz',
        'sklearn',
        'pyqlib',
        #'qlib @ git+https://github.com/microsoft/qlib@v0.8.6#egg=qlib'
    ],
    extras_require={
        'visualization': [
            'pandas-profiling>=3.3.0',
        ],
        'test': [
            'pytest~=6.2.5',
            'pytest-cov~=3.0.0',
            'pytest-mock~=3.6.1',
            'pytest-timeout~=2.0.2',
            'pytest-xdist>=1.34.0',
            'pytest-benchmark~=3.4.0',
        ]
    },
    classifiers=[
        'Development Status :: 1 - Planning',
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
