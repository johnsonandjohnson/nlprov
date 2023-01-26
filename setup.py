"""
Copyright © 2020 Johnson & Johnson
"""

from setuptools import setup, find_packages

pkgs = find_packages()
readme = open('README.md', encoding='utf-8').read()

setup(
    name='nlprov',
    version='1.1.0',
    description='A Python package for common Natural Language Processing tasks',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Johnson & Johnson HTC Advanced Analytics',
    url='https://github.com/johnsonandjohnson/nlprov',
    packages=pkgs,
    install_requires=[
        'pandas>=1.0.0',
        'spacy>=3.4',
        'nltk>=3.4.3',
        'langid>=1.1.6',
        'scikit-learn>=0.21.3'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering'
    ],
    tests_require=['pytest'],
    python_requires='>=3.7'
)


# https://github.com/tomchen/example_pypi_package/blob/d50c61f1317c0cc7d1fc8927d666da7355732c61/README.md
