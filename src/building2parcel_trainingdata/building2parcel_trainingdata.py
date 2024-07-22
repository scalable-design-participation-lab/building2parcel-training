"""
This module provides functionality for mapping parcels and buildings using various data sources and visualization methods.

The main class, Building2ParcelMapper, handles loading, processing, and visualizing geospatial data for parcels and buildings.
"""
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
    """
    A class for mapping parcels and buildings using various data sources and visualization methods.

    This class provides methods for loading geospatial data, processing it, and creating visualizations
    of parcels and buildings using different mapping techniques.

    Attributes:
        parcel_path (str): Path to the parcels shapefile or geoJSON.
        buildings_path (str): Path to the buildings shapefile or geoJSON.
        blocks_path (str): Path to the blocks shapefile or geoJSON (optional).
        epsg (int): EPSG code for the coordinate reference system (default is 3857).
        df_parcels (GeoDataFrame): GeoDataFrame containing parcel data.
        df_buildings (GeoDataFrame): GeoDataFrame containing building data.
        df_blocks (GeoDataFrame): GeoDataFrame containing block data (if provided).
        buildings_with_parcel_info (GeoDataFrame): GeoDataFrame containing joined parcel and building data.

    """

    def __init__(self, parcel_path, buildings_path, blocks_path=None, epsg=3857):
        """
        Initialize the Building2ParcelMapper with paths to parcel and building data.

        Args:
            parcel_path (str): Path to the parcels shapefile or geoJSON.
            buildings_path (str): Path to the buildings shapefile or geoJSON.
            blocks_path (str, optional): Path to the blocks shapefile or geoJSON. Defaults to None.
            epsg (int, optional): EPSG code for the coordinate reference system. Defaults to 3857.
        """

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
        """
        Load parcel, building, and block data from files and prepare it for mapping.

        This method reads the data, transforms it to the specified coordinate system,
        removes duplicates, and joins the parcel and building data.
        """

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
        
        # Assign colors to parcels
        self.df_parcels['color'] = [self.random_hex_color() for _ in range(len(self.df_parcels))]
        
        self.buildings_with_parcel_info = self.df_buildings.sjoin(self.df_parcels, how="inner")

    def remove_duplicates(self, df):
        """
        Remove duplicate geometries from a GeoDataFrame.

        Args:
            df (GeoDataFrame): The GeoDataFrame to remove duplicates from.

        Returns:
            GeoDataFrame: A new GeoDataFrame with duplicate geometries removed.
        """

        return df.drop_duplicates(subset='geometry').reset_index(drop=True)

    def random_hex_color(self, seed=None):
        """
        Generate a random hex color code.

        Args:
            seed (int, optional): Seed for the random number generator. Defaults to None.

        Returns:
            str: A randomly generated hex color code.
        """

        if seed:
            random.seed(seed)
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def split_buildings(self, threshold_high=0.75, threshold_low=0.15):
        """
        Split buildings that span multiple parcels based on overlap thresholds.

        Args:
            threshold_high (float, optional): High threshold for building-parcel overlap. Defaults to 0.75.
            threshold_low (float, optional): Low threshold for building-parcel overlap. Defaults to 0.15.
        """

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
        """
        Assign random colors to parcels and associated buildings for visualization.
        """

        self.df_parcels['color'] = [self.random_hex_color() for _ in range(len(self.df_parcels))]
        color_map = self.df_parcels.set_index('parcel_id')['color'].to_dict()
        self.buildings_with_parcel_info['color'] = self.buildings_with_parcel_info['parcel_id'].map(color_map)

    def generate_dataset_specs(self, output_folder='./dataset_specs'):
        """
        Generate and save dataset specifications and statistics.

        Args:
            output_folder (str, optional): Directory to save the output files. Defaults to './dataset_specs'.
        """

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
        """
        Create a subset of data around a selected feature.

        Args:
            index (int): Index of the feature to create a subset around.
            distance (float, optional): Buffer distance in meters. Defaults to 75.

        Returns:
            tuple: A tuple containing:
                - GeoDataFrame: Subset of parcel data.
                - GeoDataFrame: Subset of building data.
                - tuple: Bounding box of the subset area.
        """

        selected_feature = self.df_parcels.loc[index]
        geometry_buffer = selected_feature.geometry.buffer(distance)
        geometry_bounds = selected_feature.geometry.buffer(distance-70)
        return (self.df_parcels[self.df_parcels.within(geometry_buffer)],
                self.buildings_with_parcel_info[self.buildings_with_parcel_info.within(geometry_buffer)],
                geometry_bounds.bounds)

    def add_geometries(self, ax, df, crs_epsg, random_color=False):
        """
        Add geometries to the given axes object.

        Args:
            ax (GeoAxesSubplot): The axes object to add geometries to.
            df (GeoDataFrame): GeoDataFrame containing geometry data.
            crs_epsg (CRS): Coordinate reference system for the geometries.
            random_color (bool, optional): Whether to use random colors. Defaults to False.
        """

        for row in df.itertuples():
            color = self.random_hex_color(int(row.building_id)) if random_color else row.color
            ax.add_geometries(row.geometry, crs=crs_epsg, facecolor=color)

    def map_maker(self, df_parcels, df_buildings, bounds, index, scale=10, feature_type='both', random_color=False, output_folder=''):
        """
        Create a map using Mapbox satellite imagery as a base layer.

        Args:
            df_parcels (GeoDataFrame): GeoDataFrame containing parcel data.
            df_buildings (GeoDataFrame): GeoDataFrame containing building data.
            bounds (tuple): Bounding box for the map (minx, miny, maxx, maxy).
            index (int): Index for the output filename.
            scale (int, optional): Zoom level for the satellite imagery. Defaults to 10.
            feature_type (str, optional): Type of features to display ('parcels', 'buildings', or 'both'). Defaults to 'both'.
            random_color (bool, optional): Whether to use random colors. Defaults to False.
            output_folder (str, optional): Folder to save the output image. Defaults to ''.
        """

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

    def generate_images(self, parcel_images_directory, buildings_images_directory, combined_images_directory, number_of_images):
        """
        Generate, save, and combine a specified number of parcel and building images.

        Args:
            parcel_images_directory (str): Directory to save parcel images.
            buildings_images_directory (str): Directory to save building images.
            combined_images_directory (str): Directory to save combined images.
            number_of_images (int): Number of images to generate.
        """
        os.makedirs(parcel_images_directory, exist_ok=True)
        os.makedirs(buildings_images_directory, exist_ok=True)
        os.makedirs(combined_images_directory, exist_ok=True)

        indices_to_print = random.sample(range(len(self.df_parcels)), number_of_images)
        
        for i in tqdm(indices_to_print, desc="Generating and combining images"):
            try:
                subset_features = self.subset(i, 200)
                
                building_image_path = os.path.join(buildings_images_directory, f'buildings_{i}.jpg')
                parcel_image_path = os.path.join(parcel_images_directory, f'parcels_{i}.jpg')
                combined_image_path = os.path.join(combined_images_directory, f'building{i}_to_parcel{i}.jpg')

                self.map_maker(subset_features[0], subset_features[1], subset_features[2], i, 18, 'parcels', output_folder=parcel_images_directory)
                self.map_maker(subset_features[0], subset_features[1], subset_features[2], i, 18, 'buildings', output_folder=buildings_images_directory)
                
                # Combine the images
                self.combine_images(building_image_path, parcel_image_path, combined_image_path)
                
            except Exception as e:
                print(f"Error at index {i}: {str(e)}")

    def combine_images(self, building_image_path, parcel_image_path, output_path):
        """
        Combine building and parcel images side by side using Pillow.

        Args:
            building_image_path (str): Path to the building image.
            parcel_image_path (str): Path to the parcel image.
            output_path (str): Path to save the combined image.
        """
        building_img = Image.open(building_image_path)
        parcel_img = Image.open(parcel_image_path)

        # Create a new image with the width of both images and the height of the taller image
        total_width = building_img.width + parcel_img.width
        max_height = max(building_img.height, parcel_img.height)
        combined_img = Image.new('RGB', (total_width, max_height))

        # Paste the images side by side
        combined_img.paste(building_img, (0, 0))
        combined_img.paste(parcel_img, (building_img.width, 0))

        # Save the combined image
        combined_img.save(output_path)