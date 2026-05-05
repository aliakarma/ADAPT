from setuptools import setup, find_packages

setup(
    name="ADAPT",
    version="1.0.0",
    description="Agentic AI Nutrition and Healthcare Monitor for Neurodivergent and Disabled Users",
    packages=find_packages(exclude=["tests*", "experiments*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "transformers>=4.35.0",
        "matplotlib>=3.8.0",
        "PyYAML>=6.0.1",
        "loguru>=0.7.2",
        "tqdm>=4.66.1",
    ],
)
