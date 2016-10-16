import numpy
from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension

extensions = [
    Extension(
        "outbrain/libffm_helpers",
        ["outbrain/libffm_helpers.pyx"],
        include_dirs=[numpy.get_include()])
    ]

setup(name='outbrain',
      version='1.0',
      description='Things for the outbrain comp',
      author='Richard Weiss',
      author_email='richardweiss@richardweiss.org',
      packages=find_packages(),
      install_requires=[
          'Keras',
          'PyYAML',
          'nose',
          'boto',
          'boto3',
          'luigi',
          'ml_metrics',
          'numpy',
          'pandas',
          'plumbum',
          'scikit-learn',
          'coloredlogs',
          'scipy',
          'sklearn',
          #'tensorflow',
          'joblib',
          'tqdm',
      ],
      include_package_data=True,
      ext_modules=cythonize(extensions),
      )


