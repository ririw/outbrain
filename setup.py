from setuptools import setup, find_packages

setup(name='outbrain',
      version='1.0',
      description='Things for the outbrain comp',
      author='Richard Weiss',
      author_email='richardweiss@richardweiss.org',
      packages=find_packages(),
      install_requires=[
        'Keras',
        'PyYAML',
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
        'tensorflow',
      ]
)
