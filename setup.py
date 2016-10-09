from setuptools import setup, find_packages

setup(name='outbrain',
      version='1.0',
      description='Things for the outbrain comp',
      author='Richard Weiss',
      author_email='richardweiss@richardweiss.org',
      packages=find_packages(),
      install_requires=[
          'pandas',
          'plumbum',
          'luigi',
          'sklearn',
          'scipy',
          'numpy',
          'tensorflow',
          'keras',
          'boto',
          'boto3',
          'ml_metrics'
      ],
      entry_points=[]
      )
