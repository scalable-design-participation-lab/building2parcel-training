# building2parcel-training

building2parcel-training is a Python package for mapping parcels and buildings, designed to assist in training models to associate buildings with their corresponding parcels. It provides functionality for loading, processing, and visualizing geospatial data for parcels and buildings.

## Features

- Load and process parcel and building data from shapefiles or geoJSON
- Optional loading and processing of block data
- Join parcel and building data based on spatial relationships
- Split buildings that span multiple parcels
- Generate maps using Mapbox satellite imagery
- Customize map output with various options
- Generate dataset specifications and statistics
- Support for creating training datasets for building-to-parcel association models

## Installation

You can install building2parcel-training using pip:

```
pip install building2parcel-trainingdata
```

For development, clone the repository and install in editable mode:

```
git clone https://github.com/scalable-design-participation-lab/building2parcel-trainingdata.git
cd building2parcel-trainingdata
pip install -e .
```

## Configuration

Before using the package, you need to set up your environment:

1. Create a `.env` file in the main folder (it will be a hidden file on Unix-based systems).
2. In the `.env` file, add the following lines:

```
MAPBOX_ACCESS_TOKEN="YOUR-API-KEY"
LOCAL_PATH="YOUR-DROPBOX-PATH/Million Neighborhoods/"
```

Replace `YOUR-API-KEY` with your Mapbox Access Token for the Mapbox Web API, and `YOUR-DROPBOX-PATH` with the path to your Dropbox folder containing the parcel and building data (NYC data is available on our Dropbox).

## Usage

Here's a basic example of how to use building2parcel-training:

```python
from building2parcel_trainingdata import Building2ParcelMapper

# Initialize the mapper with paths to your data
parcels_path = "path/to/your/parcels.shp"
buildings_path = "path/to/your/buildings.shp"
blocks_path = "path/to/your/blocks.shp"  # Optional
mapper = Building2ParcelMapper(parcels_path, buildings_path, blocks_path)

# Split buildings (optional)
mapper.split_buildings(threshold_high=0.75, threshold_low=0.15)

# Assign colors to parcels and buildings
mapper.assign_colors()

# Generate dataset specifications and statistics
mapper.generate_dataset_specs(output_folder='./dataset_specs')

# Generate images
parcel_images_directory = "./parcels_output/"
buildings_images_directory = "./buildings_output/"
number_of_images = 100
mapper.generate_images(parcel_images_directory, buildings_images_directory, number_of_images)
```

## Command-line Usage

The package also provides a command-line interface:

```
python -m building2parcel_trainingdata --buildings_path path/to/buildings.shp --parcels_path path/to/parcels.shp --blocks_path path/to/blocks.shp --split_buildings True --threshold_high 0.75 --threshold_low 0.15 --parcel_images_directory ./parcels_output/ --buildings_images_directory ./buildings_output/ --number_of_images 100
```

## Requirements

- matplotlib
- cartopy
- geopandas
- python-dotenv
- Pillow
- numpy
- owslib
- tqdm
- pandas

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to Mapbox for providing satellite imagery services.
- This project was developed to support machine learning efforts in associating buildings with their corresponding parcels.
