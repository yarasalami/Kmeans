from distutils.core import setup, Extension

setup(
    name='mykmeanssp',
    author='Yara and Eldad',
    version='1.0',
    ext_modules=[Extension('mykmeanssp',sources=['kmeans.c'])])