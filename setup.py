from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='ReplayTables-andnp',
    url='https://github.com/andnp/ReplayTables.git',
    author='Andy Patterson',
    author_email='andnpatterson@gmail.com',
    packages=find_packages(exclude=['tests*']),
    version='0.0.2',
    license='MIT',
    description='A simple replay buffer implementation in python for sampling n-step trajectories',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=[
        'numpy>=1.21.0',
        'numba>=0.55.0',
    ],
    extras_require={},
)
