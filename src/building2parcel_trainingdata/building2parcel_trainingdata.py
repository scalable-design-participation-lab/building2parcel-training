# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
# import cartopy.crs as ccrs
# from cartopy.io.img_tiles import MapboxTiles
# import geopandas as gpd
# import os
# from dotenv import load_dotenv
# import random
# from urllib.request import urlopen
# from PIL import Image
# import numpy as np
# from owslib.wms import WebMapService

# """
# This module provides functionality for mapping parcels and buildings using various data sources and visualization methods.

# The main class, ParcelBuildingMapper, handles loading, processing, and visualizing geospatial data for parcels and buildings.
# """

# class ParcelBuildingMapper:
#     """
#     A class for mapping parcels and buildings using various data sources and visualization methods.

#     This class provides methods for loading geospatial data, processing it, and creating visualizations
#     of parcels and buildings using different mapping techniques.

#     Attributes:
#         parcels_path (str): Path to the parcels shapefile.
#         buildings_path (str): Path to the buildings shapefile.
#         epsg (int): EPSG code for the coordinate reference system (default is 3857).
#         df_parcels (GeoDataFrame): GeoDataFrame containing parcel data.
#         df_buildings (GeoDataFrame): GeoDataFrame containing building data.
#         df_parcels_buildings (GeoDataFrame): GeoDataFrame containing joined parcel and building data.

#     """
     
#     def __init__(self, parcels_path, buildings_path, epsg=3857):
#         """
#         Initialize the ParcelBuildingMapper with paths to parcel and building data.

#         Args:
#             parcels_path (str): Path to the parcels shapefile.
#             buildings_path (str): Path to the buildings shapefile.
#             epsg (int, optional): EPSG code for the coordinate reference system. Defaults to 3857.
#         """

#         load_dotenv()
#         self.parcels_path = parcels_path
#         self.buildings_path = buildings_path
#         self.epsg = epsg
#         self.df_parcels = None
#         self.df_buildings = None
#         self.df_parcels_buildings = None
#         self.load_data()

#     def load_data(self):
#         """
#         Load parcel and building data from shapefiles and prepare it for mapping.

#         This method reads the shapefile data, transforms it to the specified coordinate system,
#         assigns random colors to parcels, and joins the parcel and building data.
#         """

#         self.df_parcels = gpd.read_file(self.parcels_path)
#         self.df_buildings = gpd.read_file(self.buildings_path)
        
#         self.df_parcels = self.df_parcels.to_crs(epsg=self.epsg)
#         self.df_buildings = self.df_buildings.to_crs(epsg=self.epsg)
        
#         self.df_parcels['color'] = [self.random_hex_color() for _ in range(len(self.df_parcels))]
#         self.df_parcels_buildings = self.join_parcels_buildings(self.df_parcels, self.df_buildings)

#     def random_hex_color(self, use_seed=False):
#         """
#         Generate a random hex color code.

#         Args:
#             use_seed (bool, optional): Whether to use a seed for random number generation. Defaults to False.

#         Returns:
#             str: A randomly generated hex color code.
#         """

#         if use_seed:
#             random.seed(use_seed)
#             r = random.randint(0, 255)
#             random.seed(use_seed+1000)
#             g = random.randint(0, 255)
#             random.seed(use_seed+2000)
#             b = random.randint(0, 255)
#         else:
#             r = random.randint(0, 255)
#             g = random.randint(0, 255)
#             b = random.randint(0, 255)
#         return "#{:02x}{:02x}{:02x}".format(r, g, b)

#     def join_parcels_buildings(self, parcels, buildings):
#         """
#         Join parcel and building data based on spatial relationship.

#         Args:
#             parcels (GeoDataFrame): GeoDataFrame containing parcel data.
#             buildings (GeoDataFrame): GeoDataFrame containing building data.

#         Returns:
#             GeoDataFrame: A GeoDataFrame containing joined parcel and building data.
#         """

#         return buildings.sjoin(parcels, how="inner")

#     def add_geometries(self, ax, df_parcels, crs_epsg, random_color=False):
#         """
#         Add geometries to the given axes object.

#         Args:
#             ax (GeoAxesSubplot): The axes object to add geometries to.
#             df_parcels (GeoDataFrame): GeoDataFrame containing parcel data.
#             crs_epsg (CRS): Coordinate reference system for the geometries.
#             random_color (bool, optional): Whether to use random colors. Defaults to False.
#         """

#         for row in df_parcels.itertuples():
#             geometry = row.geometry
#             if random_color:
#                 color = self.random_hex_color(int(row.bin))
#             else:
#                 color = row.color
#             ax.add_geometries(geometry, crs=crs_epsg, facecolor=color)

#     def map_maker_mapbox_satellite(self, df_parcels, df_buildings, bounds, index, scale=10, feature_type='both', random_color=False, output_folder=''):
#         """
#         Create a map using Mapbox satellite imagery as a base layer.

#         Args:
#             df_parcels (GeoDataFrame): GeoDataFrame containing parcel data.
#             df_buildings (GeoDataFrame): GeoDataFrame containing building data.
#             bounds (tuple): Bounding box for the map (minx, miny, maxx, maxy).
#             index (int): Index for the output filename.
#             scale (int, optional): Zoom level for the satellite imagery. Defaults to 10.
#             feature_type (str, optional): Type of features to display ('parcels', 'buildings', or 'both'). Defaults to 'both'.
#             random_color (bool, optional): Whether to use random colors. Defaults to False.
#             output_folder (str, optional): Folder to save the output image. Defaults to ''.
#         """

#         access_token = os.environ.get('MAPBOX_ACCESS_TOKEN')
#         tiler = MapboxTiles(access_token, 'satellite-v9')
#         crs_epsg = ccrs.epsg(str(self.epsg))
#         mercator = tiler.crs

#         fig = plt.figure(figsize=(7, 7), dpi=96)
#         ax = fig.add_subplot(1, 1, 1, projection=mercator)

#         dist1 = bounds[2] - bounds[0]
#         dist2 = bounds[3] - bounds[1]
#         max_dist = max(dist1, dist2) / 2
#         centroid_x = (bounds[2] + bounds[0]) / 2
#         centroid_y = (bounds[3] + bounds[1]) / 2

#         ax.set_extent([centroid_x-max_dist, centroid_x+max_dist, centroid_y-max_dist, centroid_y+max_dist], crs=crs_epsg)

#         if feature_type in ['parcels', 'both']:
#             self.add_geometries(ax, df_parcels, crs_epsg, random_color)
#         if feature_type in ['buildings', 'both']:
#             self.add_geometries(ax, df_buildings, crs_epsg, random_color)

#         ax.add_image(tiler, scale)

#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)

#         plt.savefig(os.path.join(output_folder, f'{feature_type}_{index}.jpg'), bbox_inches='tight', pad_inches=0, dpi=96)
#         plt.close(fig)

#     def map_maker_simple(self, df_parcels, df_buildings, bounds, index, feature_type='both', random_color=False, output_folder=''):
#         """
#         Create a simple map without satellite imagery.

#         Args:
#             df_parcels (GeoDataFrame): GeoDataFrame containing parcel data.
#             df_buildings (GeoDataFrame): GeoDataFrame containing building data.
#             bounds (tuple): Bounding box for the map (minx, miny, maxx, maxy).
#             index (int): Index for the output filename.
#             feature_type (str, optional): Type of features to display ('parcels', 'buildings', or 'both'). Defaults to 'both'.
#             random_color (bool, optional): Whether to use random colors. Defaults to False.
#             output_folder (str, optional): Folder to save the output image. Defaults to ''.
#         """

#         fig, ax = plt.subplots(figsize=(7, 7))
        
#         if feature_type in ['parcels', 'both']:
#             df_parcels.plot(ax=ax, facecolor=df_parcels['color'], edgecolor='black', linewidth=0.5)
#         if feature_type in ['buildings', 'both']:
#             df_buildings.plot(ax=ax, facecolor='red', edgecolor='black', linewidth=0.5)
        
#         ax.set_xlim(bounds[0], bounds[2])
#         ax.set_ylim(bounds[1], bounds[3])
#         ax.axis('off')
        
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
        
#         plt.savefig(os.path.join(output_folder, f'{feature_type}_{index}.jpg'), bbox_inches='tight', pad_inches=0, dpi=96)
#         plt.close(fig)

#     def map_maker_nasa_gibs_rest(self, df_parcels, df_buildings, bounds, index, scale=10, feature_type='both', random_color=False, output_folder=''):
#         """
#         Create a map using NASA GIBS REST API for satellite imagery as a base layer.

#         This method generates a map using a single tile from the NASA GIBS REST API as the base layer,
#         and overlays parcel and/or building data on top of it.

#         Args:
#             df_parcels (GeoDataFrame): GeoDataFrame containing parcel data.
#             df_buildings (GeoDataFrame): GeoDataFrame containing building data.
#             bounds (tuple): Bounding box for the map (minx, miny, maxx, maxy).
#             index (int): Index for the output filename.
#             scale (int, optional): Zoom level for the satellite imagery. Defaults to 10.
#             feature_type (str, optional): Type of features to display ('parcels', 'buildings', or 'both'). Defaults to 'both'.
#             random_color (bool, optional): Whether to use random colors for features. Defaults to False.
#             output_folder (str, optional): Folder to save the output image. Defaults to ''.

#         Returns:
#             None

#         Note:
#             This method saves the generated map as a JPEG file in the specified output folder.
#             The filename format is '{feature_type}_{index}.jpg'.
#         """

#         crs_epsg = ccrs.epsg(str(self.epsg))
        
#         layer = "MODIS_Terra_CorrectedReflectance_TrueColor"
#         date = "2020-03-01"  # Example date, adjust as necessary
#         zoom_level = 6  # Zoom level
#         tile_row = 10  # Tile row
#         tile_col = 21  # Tile column
#         tile_url = f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{date}/GoogleMapsCompatible_Level9/{zoom_level}/{tile_row}/{tile_col}.jpg"
        
#         fig = plt.figure(figsize=(8, 8))
#         ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())

#         dist1 = bounds[2] - bounds[0]
#         dist2 = bounds[3] - bounds[1]
#         max_dist = max(dist1, dist2) / 2
#         centroid_x = (bounds[2] + bounds[0]) / 2
#         centroid_y = (bounds[3] + bounds[1]) / 2

#         ax.set_extent([centroid_x-max_dist, centroid_x+max_dist, centroid_y-max_dist, centroid_y+max_dist], crs=crs_epsg)

#         with urlopen(tile_url) as url:
#             img = Image.open(url)
#             img_array = np.array(img)
#         img_extent = [-130, -100, 20, 50]  # Adjust this extent to match the tile's coverage
#         ax.imshow(img_array, origin='upper', extent=img_extent, transform=ccrs.PlateCarree())
        
#         if feature_type in ['parcels', 'both']:
#             self.add_geometries(ax, df_parcels, crs_epsg, random_color)
#         if feature_type in ['buildings', 'both']:
#             self.add_geometries(ax, df_buildings, crs_epsg, random_color)

#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
#         plt.savefig(os.path.join(output_folder, f'{feature_type}_{index}.jpg'), bbox_inches='tight', pad_inches=0)
#         plt.close(fig)

#     def map_maker_nasa_gibs_wms(self, df_parcels, df_buildings, bounds, index, scale=10, feature_type='both', random_color=False, output_folder=''):
#         """
#         Create a map using NASA GIBS Web Map Service (WMS) for satellite imagery as a base layer.

#         This method generates two maps:
#         1. A base map using the NASA GIBS WMS satellite imagery.
#         2. An overlay map with parcel and/or building data on top of the satellite imagery.

#         Args:
#             df_parcels (GeoDataFrame): GeoDataFrame containing parcel data.
#             df_buildings (GeoDataFrame): GeoDataFrame containing building data.
#             bounds (tuple): Bounding box for the map (minx, miny, maxx, maxy).
#             index (int): Index for the output filename.
#             scale (int, optional): Zoom level for the satellite imagery. Defaults to 10.
#             feature_type (str, optional): Type of features to display ('parcels', 'buildings', or 'both'). Defaults to 'both'.
#             random_color (bool, optional): Whether to use random colors for features. Defaults to False.
#             output_folder (str, optional): Folder to save the output images. Defaults to ''.

#         Returns:
#             None

#         Note:
#             This method saves two JPEG files in the specified output folder:
#             1. '{feature_type}_{index}.jpg': The base satellite image.
#             2. '{feature_type}_{index}_with_features.jpg': The satellite image with overlaid features.
#         """

#         wms = WebMapService('https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?', version='1.1.1')

#         img = wms.getmap(layers=['MODIS_Terra_CorrectedReflectance_TrueColor'],
#                         srs='epsg:4326',
#                         bbox=(-180,-90,180,90),
#                         size=(1200, 600),
#                         time='2024-01-01',
#                         format='image/jpeg',
#                         transparent=False)

#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
#         out = open(os.path.join(output_folder, f'{feature_type}_{index}.jpg'), 'wb')
#         out.write(img.read())
#         out.close()

#         fig, ax = plt.subplots(figsize=(12, 6))
#         img = Image.open(os.path.join(output_folder, f'{feature_type}_{index}.jpg'))
#         ax.imshow(img)

#         if feature_type in ['parcels', 'both']:
#             df_parcels.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=0.5)
#         if feature_type in ['buildings', 'both']:
#             df_buildings.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.5)

#         ax.set_xlim(0, 1200)
#         ax.set_ylim(600, 0)
#         ax.axis('off')

#         plt.savefig(os.path.join(output_folder, f'{feature_type}_{index}_with_features.jpg'), bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
    
#     def subset(self, df, df_buildings, index, distance=75):
#         """
#         Create a subset of the data based on a buffer around a selected feature.

#         Args:
#             df (GeoDataFrame): GeoDataFrame containing parcel data.
#             df_buildings (GeoDataFrame): GeoDataFrame containing building data.
#             index (int): Index of the feature to create a subset around.
#             distance (float, optional): Buffer distance in meters. Defaults to 75.

#         Returns:
#             tuple: A tuple containing:
#                 - GeoDataFrame: Subset of parcel data.
#                 - GeoDataFrame: Subset of building data.
#                 - tuple: Bounding box of the subset area.
#         """

#         selected_feature = df.loc[index]
#         geometry_buffer = selected_feature.geometry.buffer(distance)
#         geometry_bounds = selected_feature.geometry.buffer(distance-70)
#         return df[df.within(geometry_buffer)], df_buildings[df_buildings.within(geometry_buffer)], geometry_bounds.bounds

#     def generate_maps(self, parcels_output_path, buildings_output_path, start_index=0, end_index=10, distance=75, map_type='mapbox_satellite'):
#         """
#         Generate maps for a range of indices using the specified map type.

#         Args:
#             parcels_output_path (str): Output folder for parcel maps.
#             buildings_output_path (str): Output folder for building maps.
#             start_index (int, optional): Starting index for map generation. Defaults to 0.
#             end_index (int, optional): Ending index for map generation. Defaults to 10.
#             distance (float, optional): Buffer distance in meters for subsetting. Defaults to 75.
#             map_type (str, optional): Type of map to generate. Defaults to 'mapbox_satellite'.
#                 Supported types:
#                 - 'mapbox_satellite': Uses Mapbox satellite imagery.
#                 - 'simple': Creates a simple map without satellite imagery.
#                 - 'nasa_gibs_rest': Uses NASA GIBS REST API for satellite imagery.
#                 - 'nasa_gibs_wms': Uses NASA GIBS WMS for satellite imagery.

#         Raises:
#             ValueError: If an unsupported map type is specified.

#         Note:
#             This method generates multiple maps based on the specified parameters and map type.
#             The output files are saved in the provided output folders.
#         """
        
#         for i in range(start_index, end_index):
#             subset_features = self.subset(self.df_parcels, self.df_parcels_buildings, i, distance)
            
#             if map_type == 'mapbox_satellite':
#                 self.map_maker_mapbox_satellite(subset_features[0], subset_features[1], subset_features[2], i, 18, 'buildings', output_folder=buildings_output_path)
#                 self.map_maker_mapbox_satellite(subset_features[0], subset_features[1], subset_features[2], i, 18, 'parcels', output_folder=parcels_output_path)
#             elif map_type == 'simple':
#                 self.map_maker_simple(subset_features[0], subset_features[1], subset_features[2], i, 'buildings', output_folder=buildings_output_path)
#                 self.map_maker_simple(subset_features[0], subset_features[1], subset_features[2], i, 'parcels', output_folder=parcels_output_path)
#             elif map_type == 'nasa_gibs_rest':
#                 self.map_maker_nasa_gibs_rest(subset_features[0], subset_features[1], subset_features[2], i, 18, 'both', output_folder=buildings_output_path)
#             elif map_type == 'nasa_gibs_wms':
#                 self.map_maker_nasa_gibs_wms(subset_features[0], subset_features[1], subset_features[2], i, 18, 'both', output_folder=buildings_output_path)
#             else:
#                 raise ValueError(f"Unsupported map type: {map_type}")
#         plt.close('all')

# if __name__ == "__main__":
#     parcels_path = "C:/Million Neighborhoods/Spatial Data/ny-manhattan-parcels/NYC_2021_Tax_Parcels_SHP_2203/NewYork_2021_Tax_Parcels_SHP_2203.shp"
#     buildings_path = "C:/Million Neighborhoods/Spatial Data/ny-manhattan-buildings/geo_export_a80ea1a2-e8e0-4ffd-862c-1199433ac303.shp"
#     mapper = ParcelBuildingMapper(parcels_path, buildings_path)
    
#     parcels_output_path = "C:/Million Neighborhoods/Spatial Data/result/parcels/"
#     buildings_output_path = "C:/Million Neighborhoods/Spatial Data/result/buildings/"
    
#     # Generate maps using Mapbox satellite imagery
#     mapper.generate_maps(parcels_output_path, buildings_output_path, start_index=0, end_index=5, distance=200, map_type='mapbox_satellite')

#     # Generate maps using NASA GIBS REST API
#     mapper.generate_maps(parcels_output_path, buildings_output_path, start_index=10, end_index=15, distance=200, map_type='nasa_gibs_rest')
    
#     # Generate simple maps without satellite imagery
#     mapper.generate_maps(parcels_output_path, buildings_output_path, start_index=5, end_index=10, distance=200, map_type='simple')

#     # Generate maps using NASA GIBS WMS
#     mapper.generate_maps(parcels_output_path, buildings_output_path, start_index=15, end_index=20, distance=200, map_type='nasa_gibs_wms')
    

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cartopy.crs as ccrs
from cartopy.io.img_tiles import MapboxTiles
import geopandas as gpd
import os
from dotenv import load_dotenv
import random
from urllib.request import urlopen
from PIL import Image
import numpy as np
from owslib.wms import WebMapService
import argparse
from tqdm import tqdm
import pandas as pd

class Building2ParcelMapper:
    def __init__(self, parcel_path, buildings_path, blocks_path=None, epsg=3857):
        load_dotenv(override=True)
        self.parcel_path = parcel_path
        self.buildings_path = buildings_path
        self.blocks_path = blocks_path
        self.epsg = epsg
        self.df_parcels = None
        self.df_buildings = None
        self.df_blocks = None
        self.buildings_with_parcel_info = None
        self.load_data()

    def load_data(self):
        self.df_parcels = gpd.read_file(self.parcel_path).to_crs(epsg=self.epsg)
        self.df_buildings = gpd.read_file(self.buildings_path).to_crs(epsg=self.epsg)
        
        if self.blocks_path:
            self.df_blocks = gpd.read_file(self.blocks_path).to_crs(epsg=self.epsg)
            self.df_blocks = self.remove_duplicates(self.df_blocks)
            self.df_parcels = gpd.sjoin(self.df_parcels, self.df_blocks, op='within')
            self.df_parcels = self.df_parcels.drop(columns=['index_right']).reset_index(drop=True)

        self.df_parcels = self.remove_duplicates(self.df_parcels)
        self.df_buildings = self.remove_duplicates(self.df_buildings)

        self.df_buildings['building_id'] = range(len(self.df_buildings))
        self.df_parcels['parcel_id'] = range(len(self.df_parcels))
        
        self.buildings_with_parcel_info = self.df_buildings.sjoin(self.df_parcels, how="inner")

    def remove_duplicates(self, df):
        return df.drop_duplicates(subset='geometry').reset_index(drop=True)

    def random_hex_color(self, seed=None):
        if seed:
            random.seed(seed)
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def split_buildings(self, threshold_high=0.75, threshold_low=0.15):
        building_counts = self.buildings_with_parcel_info.groupby("building_id").size()
        buildings_with_multiple_parcels = building_counts[building_counts > 1].index.tolist()
        
        split_buildings = gpd.GeoDataFrame()
        to_remove = {}

        for building_id in tqdm(buildings_with_multiple_parcels, desc="Splitting buildings"):
            building = self.df_buildings[self.df_buildings["building_id"] == building_id].copy()
            parcel_ids = self.buildings_with_parcel_info[self.buildings_with_parcel_info["building_id"] == building_id]["parcel_id"].tolist()

            split_geometries = []
            areas = []
            
            for parcel_id in parcel_ids:
                split_geometry = self.df_parcels[self.df_parcels["parcel_id"] == parcel_id].geometry.intersection(building.geometry.union_all())
                split_geometries.append(split_geometry)
                areas.append(split_geometry.area.values[0])

            areas_normalized = [a/sum(areas) for a in areas]
            max_value = max(areas_normalized)
            max_index = areas_normalized.index(max_value)

            if max_value >= threshold_high:
                to_remove[building_id] = parcel_ids[:max_index] + parcel_ids[max_index + 1:]
            else:
                to_remove[building_id] = [parcel_id for i, parcel_id in enumerate(parcel_ids) if areas_normalized[i] <= threshold_low or threshold_low < areas_normalized[i] < threshold_high]
                
                for i, parcel_id in enumerate(parcel_ids):
                    if threshold_low < areas_normalized[i] < threshold_high:
                        building_temp = self.buildings_with_parcel_info[(self.buildings_with_parcel_info["building_id"] == building_id) & (self.buildings_with_parcel_info["parcel_id"] == parcel_id)].copy()
                        building_temp.geometry = split_geometries[i]
                        building_temp['building_id'] = f"{building_temp['building_id'].values[0]}_{i}"
                        split_buildings = pd.concat([split_buildings, building_temp], ignore_index=True)

        self.buildings_with_parcel_info = self.buildings_with_parcel_info[
            ~((self.buildings_with_parcel_info['building_id'].isin(to_remove.keys())) & 
              (self.buildings_with_parcel_info.apply(lambda row: row['parcel_id'] in to_remove.get(row['building_id'], []), axis=1)))
        ]
        self.buildings_with_parcel_info = pd.concat([self.buildings_with_parcel_info, split_buildings], ignore_index=True)
        self.buildings_with_parcel_info['building_id'] = self.buildings_with_parcel_info['building_id'].astype(str)

    def assign_colors(self):
        self.df_parcels['color'] = [self.random_hex_color() for _ in range(len(self.df_parcels))]
        color_map = self.df_parcels.set_index('parcel_id')['color'].to_dict()
        self.buildings_with_parcel_info['color'] = self.buildings_with_parcel_info['parcel_id'].map(color_map)

    def generate_dataset_specs(self, output_folder='./dataset_specs'):
        os.makedirs(output_folder, exist_ok=True)
        
        buildings_before = len(self.df_buildings)
        buildings_after = len(self.buildings_with_parcel_info)
        number_of_parcels = len(self.df_parcels)

        # Remove 'index_right' column if it exists
        if 'index_right' in self.buildings_with_parcel_info.columns:
            self.buildings_with_parcel_info = self.buildings_with_parcel_info.drop(columns=['index_right'])
        if 'index_right' in self.df_parcels.columns:
            self.df_parcels = self.df_parcels.drop(columns=['index_right'])

        joined_df = gpd.sjoin(self.buildings_with_parcel_info, self.df_parcels, predicate='within', how='left', rsuffix='_right')
        
        print("Columns in joined_df:", joined_df.columns)
        
        # Choose an appropriate groupby column
        possible_columns = ['parcel_id', 'parcel_id_right', 'BBL', 'BBL_right']
        groupby_column = next((col for col in possible_columns if col in joined_df.columns), None)
        
        if groupby_column is None:
            print("Warning: No suitable groupby column found. Using index.")
            joined_df = joined_df.reset_index()
            groupby_column = 'index'
        
        print(f"Grouping by column: {groupby_column}")
        
        building_counts_per_parcel = joined_df.groupby(groupby_column).size()
        parcels_with_multiple_buildings = building_counts_per_parcel[building_counts_per_parcel > 1]
        num_parcels_with_multiple_buildings = len(parcels_with_multiple_buildings)

        self.df_parcels['area'] = self.df_parcels.geometry.area

        plt.figure(figsize=(10, 6))
        plt.hist(self.df_parcels['area'], bins=50, color='skyblue', edgecolor='black')
        plt.title('Parcel Area Distribution')
        plt.xlabel('Area (square meters)')
        plt.ylabel('Number of Parcels')
        plt.grid(True)
        plt.savefig(f'{output_folder}/parcel_area_distribution.png', dpi=300)
        plt.close()

        area_description = self.df_parcels['area'].describe()

        # Ensure the groupby_column exists in both DataFrames
        if groupby_column not in self.df_parcels.columns:
            self.df_parcels[groupby_column] = self.df_parcels.index

        def safe_intersection(row):
            matching_parcels = self.df_parcels[self.df_parcels[groupby_column] == row[groupby_column]]
            if matching_parcels.empty:
                return 0
            return row['geometry'].intersection(matching_parcels.iloc[0].geometry).area

        joined_df['intersection_area'] = joined_df.apply(safe_intersection, axis=1)
        
        building_area_per_parcel = joined_df.groupby(groupby_column)['intersection_area'].sum()

        self.df_parcels['parcel_area'] = self.df_parcels.geometry.area
        self.df_parcels = self.df_parcels.merge(building_area_per_parcel, on=groupby_column, how='left')
        self.df_parcels['building_area'] = self.df_parcels['intersection_area'].fillna(0)
        self.df_parcels['coverage_percentage'] = (self.df_parcels['building_area'] / self.df_parcels['parcel_area']) * 100

        coverage_description = self.df_parcels['coverage_percentage'].describe()

        with open(f'{output_folder}/dataset_summary_and_statistics.txt', 'w') as file:
            file.write(f'Buildings before: {buildings_before}\n')
            file.write(f'Buildings after: {buildings_after}\n')
            file.write(f'Number of parcels: {number_of_parcels}\n')
            file.write(f'Parcels with multiple buildings: {num_parcels_with_multiple_buildings}\n\n')
            file.write('Area Statistics:\n')
            file.write(area_description.to_string())
            file.write('\n\nCoverage Statistics:\n')
            file.write(coverage_description.to_string())

    def subset(self, index, distance=75):
        selected_feature = self.df_parcels.loc[index]
        geometry_buffer = selected_feature.geometry.buffer(distance)
        geometry_bounds = selected_feature.geometry.buffer(distance-70)
        return (self.df_parcels[self.df_parcels.within(geometry_buffer)],
                self.buildings_with_parcel_info[self.buildings_with_parcel_info.within(geometry_buffer)],
                geometry_bounds.bounds)

    def add_geometries(self, ax, df, crs_epsg, random_color=False):
        for row in df.itertuples():
            color = self.random_hex_color(int(row.building_id)) if random_color else row.color
            ax.add_geometries(row.geometry, crs=crs_epsg, facecolor=color)

    def map_maker(self, df_parcels, df_buildings, bounds, index, scale=10, feature_type='both', random_color=False, output_folder=''):
        access_token = os.getenv('MAPBOX_ACCESS_TOKEN')
        tiler = MapboxTiles(access_token, 'satellite-v9')
        crs_epsg = ccrs.epsg(str(self.epsg))
        mercator = tiler.crs

        fig = plt.figure(figsize=(7, 7), dpi=96)
        ax = fig.add_subplot(1, 1, 1, projection=mercator)

        dist1 = bounds[2] - bounds[0]
        dist2 = bounds[3] - bounds[1]
        max_dist = max(dist1, dist2) / 2
        centroid_x = (bounds[2] + bounds[0]) / 2
        centroid_y = (bounds[3] + bounds[1]) / 2

        ax.set_extent([centroid_x-max_dist, centroid_x+max_dist, centroid_y-max_dist, centroid_y+max_dist], crs=crs_epsg)

        if feature_type in ['parcels', 'both']:
            self.add_geometries(ax, df_parcels, crs_epsg, random_color)
        if feature_type in ['buildings', 'both']:
            self.add_geometries(ax, df_buildings, crs_epsg, random_color)

        ax.add_image(tiler, scale)

        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(os.path.join(output_folder, f'{feature_type}_{index}.jpg'), bbox_inches='tight', pad_inches=0, dpi=96)
        plt.close(fig)

    def generate_images(self, parcel_images_directory, buildings_images_directory, number_of_images):
        os.makedirs(parcel_images_directory, exist_ok=True)
        os.makedirs(buildings_images_directory, exist_ok=True)

        indices_to_print = random.sample(range(len(self.df_parcels)), number_of_images)
        
        for i in tqdm(indices_to_print, desc="Generating images"):
            try:
                subset_features = self.subset(i, 200)
                self.map_maker(subset_features[0], subset_features[1], subset_features[2], i, 18, 'parcels', output_folder=parcel_images_directory)
                self.map_maker(subset_features[0], subset_features[1], subset_features[2], i, 18, 'buildings', output_folder=buildings_images_directory)
            except Exception as e:
                print(f"Error at index {i}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate parcel and building image datasets from shapefile or geoJSON data.')
    parser.add_argument("--buildings_path", help='Path to buildings (shapefile or geoJSON)', type=str, 
                        default="C:/Million Neighborhoods/Spatial Data/ny-manhattan-buildings/geo_export_a80ea1a2-e8e0-4ffd-862c-1199433ac303.shp")
    parser.add_argument("--parcels_path", help='Path to parcels (shapefile or geoJSON)', type=str, 
                        default="C:/Million Neighborhoods/Spatial Data/ny-manhattan-parcels/NYC_2021_Tax_Parcels_SHP_2203/NewYork_2021_Tax_Parcels_SHP_2203.shp")
    parser.add_argument("--blocks_path", help='Path to blocks (shapefile or geoJSON)', type=str, default=None)
    parser.add_argument("--split_buildings", help='Whether to split buildings', type=bool, default=False)
    parser.add_argument("--threshold_high", help="High building-parcel overlap threshold", type=float, default=0.75)
    parser.add_argument("--threshold_low", help="Low building-parcel overlap threshold", type=float, default=0.15)
    parser.add_argument("--parcel_images_directory", help='Directory for parcel images', type=str, default='./parcels_test/')
    parser.add_argument("--buildings_images_directory", help='Directory for building images', type=str, default='./buildings_test/')
    parser.add_argument("--number_of_images", help='Number of images to generate', type=int, default=10)
        
    args = parser.parse_args()

    parcels = Parcels(args.parcels_path, args.buildings_path, args.blocks_path)
    
    if args.split_buildings:
        parcels.split_buildings(args.threshold_high, args.threshold_low)
    
    parcels.assign_colors()
    parcels.generate_dataset_specs()
    parcels.generate_images(args.parcel_images_directory, args.buildings_images_directory, args.number_of_images)

    print('Done!')