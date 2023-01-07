from pathlib import Path

from setuptools import setup


def read_requirements():
    with open(Path(__file__).with_name("requirements.txt")) as f:
        return f.read().splitlines()


setup(
    name="imcolorize",
    version="0.0.0",
    install_requires=read_requirements(),
)
