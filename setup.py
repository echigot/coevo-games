from setuptools import setup, find_packages

setup(
   name='coevo',
   version='0.1.0',
   author='Estelle Chigot',
   author_email='estelle.chigot@gmail.com',
   packages=find_packages(),
   license='MIT',
   description='Coevolution of Agents and Games',
   long_description=open('README.md').read(),
   install_requires=[
       "torch", "gym", "stable_baselines3", "griddly", "tqdm"
   ],
)

