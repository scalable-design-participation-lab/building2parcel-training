from setuptools import setup, find_packages

setup(
    name="building2parcel_trainingdata",
    version="0.3.2",
    author="Dezeng Kong",
    author_email="kong.dez@northeastern.edu",
    description="A package for mapping parcels and buildings using various data sources",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/scalable-design-participation-lab/building2parcel-trainingdata.git",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "matplotlib",
        "cartopy",
        "geopandas",
        "python-dotenv",
        "Pillow",
        "numpy",
        "owslib",
        "tqdm",
        "pandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'building2parcel_trainingdata=building2parcel_trainingdata.building2parcel_trainingdata:main',
        ],
    },
)