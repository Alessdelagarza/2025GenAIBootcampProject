from setuptools import setup, find_packages

setup(
    name="video-effects",
    version="0.1",
    packages=find_packages(),
    install_requires=["pandas", "openai", "scikit-learn", "numpy", "streamlit"],
)
