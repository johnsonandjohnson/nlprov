"""
Copyright © 2020 Johnson & Johnson
"""

from setuptools import setup, find_packages

pkgs = find_packages()
readme = open('README.md', encoding='utf-8').read()

setup(
    name='NLProv',
    version='1.0.0',
    description='A Python package for common Natural Language Processing tasks',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Johnson & Johnson HTC Advanced Analytics',
    url='https://github.com/johnsonandjohnson/nlprov',
    packages=pkgs,
    install_requires=[
        'pandas>=1.0.0',
        'spacy>=2.1.0,<2.1.7',
        'nltk>=3.4.3',
        'langid>=1.1.6',
        'scikit-learn>=0.21.3'
    ],
    tests_require=['pytest'],
    python_requires='>=3.7'
)
