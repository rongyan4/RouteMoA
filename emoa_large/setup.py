"""Python setup.py for emoa (large model pool) package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="emoa",
    version="0.0.1",
    description="Efficient Mixture-of-Agents (Large Model Pool)",
    long_description=read("README.md"),
    author="wangjize",
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        include=["emoa", "emoa/**/*"],
        exclude=["tests", ".github"]
    ),
    install_requires=read_requirements("requirements.txt"),
)
