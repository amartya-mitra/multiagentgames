'''
Sample setup.py:
1. https://github.com/navdeep-G/setup.py
2. https://github.com/pypa/sampleproject/blob/master/setup.py
'''
from setuptools import setup, find_packages, Command
import os

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

setup(
    name='multiagentgames',
    version='1.0',
    description='Implementation of game logic and learning algorithms in Jax for multi agent games',
    author='Anirban Laha',
    author_email='anirbanlaha@gmail.com',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'jax>=0.1.57',
        'jaxlib>=0.1.37',
        'matplotlib',
        'numpy >= 1.10',
        'scipy >= 1.0',
        'scikit-learn',
        'wandb',
        'gym',
        'progressbar',
        'torch',
        'argparse_deco',
        'argparse',
        'pandas'
    ], #external packages as dependencies
    license='MIT',
    url='https://github.com/anirbanl/multiagentgames',
    python_requires='>=3.6.0',
    cmdclass={
        'clean': CleanCommand,
    }
)