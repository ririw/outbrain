from distutils.core import setup

setup(name='outbrain',
      version='1.0',
      description='Things for the outbrain comp',
      author='Richard Weiss',
      author_email='richardweiss@richardweiss.org',
      packages=['outbrain'],
      install_requires=[
          'pandas',
          'plumbum',
          'luigi',
          'sklearn',
          'scipy',
          'numpy',
          'tensorflow',
          'keras',
      ],
      entry_points=[]
)

