from setuptools import setup, find packages
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Personal Health Dashboard for Disease Prediction",
    version="0.1",
    author="surya",
    packages=find_packages(),
    install_requires=requirements
)