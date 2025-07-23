from setuptools import setup, find_packages
import os

setup(
    name="transforms3d",
    version="1.0.0",
    description="Simulation of grasp fractured objects.",
    packages=find_packages(include=["transforms3d", "transforms3d.*"]),
    python_requires=">=3.8",
    include_package_data=True,  
)