# building2parcel-training

building2parcel-training is a Python package for mapping parcels and buildings, designed to assist in training models to associate buildings with their corresponding parcels. It provides functionality for loading, processing, and visualizing geospatial data for parcels and buildings.

## Features

- Load and process parcel and building data from shapefiles
- Join parcel and building data based on spatial relationships
- Generate maps using different base layers:
  - Mapbox satellite imagery
  - NASA GIBS REST API
  - NASA GIBS Web Map Service (WMS)
  - Simple maps without satellite imagery
- Customize map output with various options
- Support for creating training datasets for building-to-parcel association models

## Installation

You can install building2parcel-training using pip:

```

pip install building2parcel-trainingdata==0.1.0

```

For development, clone the repository and install in editable mode:

```

git clone https://github.com/yourusername/building2parcel-training.git
cd building2parcel-training
pip install -e .

```

## Usage

Here's a basic example of how to use building2parcel-training:

```python
from building2parcel_training import ParcelBuildingMapper

# Initialize the mapper with paths to your data
parcels_path = "path/to/your/parcels.shp"
buildings_path = "path/to/your/buildings.shp"
mapper = ParcelBuildingMapper(parcels_path, buildings_path)

# Set output paths
parcels_output_path = "output/parcels/"
buildings_output_path = "output/buildings/"

# Generate maps using Mapbox satellite imagery
mapper.generate_maps(parcels_output_path, buildings_output_path,
                     start_index=0, end_index=5, distance=200,
                     map_type='mapbox_satellite')

# Generate simple maps without satellite imagery
mapper.generate_maps(parcels_output_path, buildings_output_path,
                     start_index=5, end_index=10, distance=200,
                     map_type='simple')

# Generate maps using NASA GIBS REST API
mapper.generate_maps(parcels_output_path, buildings_output_path,
                     start_index=10, end_index=15, distance=200,
                     map_type='nasa_gibs_rest')

# Generate maps using NASA GIBS WMS
mapper.generate_maps(parcels_output_path, buildings_output_path,
                     start_index=15, end_index=20, distance=200,
                     map_type='nasa_gibs_wms')

## Requirements

- matplotlib
- cartopy
- geopandas
- python-dotenv
- Pillow
- numpy
- owslib

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to Mapbox and NASA GIBS for providing satellite imagery services.
- This project was developed to support machine learning efforts in associating buildings with their corresponding parcels.
```
