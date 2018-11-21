from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('bbox.pyx'))
setup(ext_modules=cythonize('cython_bbox.pyx'))
