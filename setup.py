from setuptools import setup, find_packages

setup(
    name="yourtoolkit",
    version="0.1.0",
    description="Multimodal single-cell toolkit (UMAP/Leiden, GMM-based ADT denoising, QC, visualization, etc.)",
    author="Brett",
    author_email="your_email@example.com",
    url="https://github.com/yourusername/yourtoolkit",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "scanpy",
        "muon",
        "scikit-learn",
        "joblib",
    ],
    python_requires=">=3.9",
)
