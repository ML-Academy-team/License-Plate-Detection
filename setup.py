#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='licence-plate-detection',
    version='0.1.0',
    description='Tunisian vehicules licence plate detection package',
    author='',
    author_email='',
    url='https://github.com/BHafsa/Licence-Plate-Detection',
    install_requires=['tensorflow', 'keras-ocr', ''],
    packages=find_packages(),
)

