from pathlib import Path

from setuptools import setup

setup(
    name="imcolorize",
    version="0.0.0",
    install_requires=open(Path(__file__).with_name("requirements.txt")).read().splitlines(),
)
