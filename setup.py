# Databricks notebook source
from setuptools import setup, find_packages

# Add install requirements
setup(
    author="legatdavid",
    description="Testing DBC libraries",
    name="DSutils",
    packages=find_packages(include=["DSutils", "DSutils.*"]),
    version="0.1.0",
    install_requires=['numpy', 'pandas'],
    python_requires=">=3.5",
)
