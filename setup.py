""" setup """
from setuptools import find_packages, setup

requirements = open("requirements.txt").readlines()
setup(
    name="sketched_nl2sql",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=requirements,
    zip_safe=False,
)
