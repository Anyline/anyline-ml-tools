#!/usr/bin/env python
from setuptools import setup, find_packages


setup(name='anyline_mltools',
      version='1.0.0a1',
      description='Anyline ML Packages and Libraries',
      author='Dmytro Kotsur',
      author_email='dmytro@anyline.com',
      packages=find_packages(),
      license='license.txt',
      python_requires='>=3.6',
      url='https://githsub.com/pypa/sampleproject',
      install_requires=[
            "tensorflow_gpu>=2.1",
            "tensorflow_addons",
            "numpy"
      ])