import unittest
import geopandas as gpd
from shapely.geometry import Polygon
from unittest.mock import patch, MagicMock
from building2parcel_trainingdata import Building2ParcelMapper
from PIL import Image
import numpy as np
import os

class TestBuilding2ParcelMapper(unittest.TestCase):

    def setUp(self):
        # Create mock data for testing with a CRS
        self.mock_parcel_data = gpd.GeoDataFrame({
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            'parcel_id': [1]
        }, crs="EPSG:4326")
        self.mock_building_data = gpd.GeoDataFrame({
            'geometry': [Polygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])],
            'building_id': [1]
        }, crs="EPSG:4326")
        
        # Create a Building2ParcelMapper instance with mock data
        with patch('geopandas.read_file') as mock_read_file:
            mock_read_file.side_effect = [self.mock_parcel_data, self.mock_building_data]
            self.mapper = Building2ParcelMapper('mock_parcel.shp', 'mock_building.shp')

        # Override the CRS transformation in the load_data method
        self.mapper.df_parcels = self.mock_parcel_data.to_crs(epsg=3857)
        self.mapper.df_buildings = self.mock_building_data.to_crs(epsg=3857)
        self.mapper.buildings_with_parcel_info = gpd.sjoin(self.mapper.df_buildings, self.mapper.df_parcels, how="inner", predicate='intersects')

    def test_remove_duplicates(self):
        # Create a GeoDataFrame with duplicate geometries
        df = gpd.GeoDataFrame({
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                         Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                         Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])],
            'value': [1, 2, 3]
        }, crs="EPSG:4326")
        
        result = self.mapper.remove_duplicates(df)
        self.assertEqual(len(result), 2)
        self.assertListEqual(list(result['value']), [1, 3])
        

    def test_random_hex_color(self):
        color1 = self.mapper.random_hex_color()
        color2 = self.mapper.random_hex_color()
        
        # Check if the colors are valid hex colors
        self.assertTrue(color1.startswith('#') and len(color1) == 7)
        self.assertTrue(color2.startswith('#') and len(color2) == 7)
        
        # Check if two generated colors are different (this might rarely fail)
        self.assertNotEqual(color1, color2)
        

    @patch('random.randint')
    def test_random_hex_color_with_seed(self, mock_randint):
        mock_randint.return_value = 0xABCDEF
        color = self.mapper.random_hex_color(seed=42)
        self.assertEqual(color, "#abcdef")
        

    def test_split_buildings(self):
        # Create a building that spans two parcels
        building = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
        parcel1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        parcel2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
        
        # Set up the test data
        self.mapper.df_buildings = gpd.GeoDataFrame({
            'geometry': [building],
            'building_id': [1]
        }, crs="EPSG:3857")
        self.mapper.df_parcels = gpd.GeoDataFrame({
            'geometry': [parcel1, parcel2],
            'parcel_id': [1, 2]
        }, crs="EPSG:3857")
        self.mapper.buildings_with_parcel_info = gpd.GeoDataFrame({
            'geometry': [building, building],
            'building_id': [1, 1],
            'parcel_id': [1, 2]
        }, crs="EPSG:3857")
        
        # Execute the building split
        self.mapper.split_buildings()
        
        # Assign colors
        self.mapper.assign_colors()
        
        # Check the results
        self.assertEqual(len(self.mapper.buildings_with_parcel_info), 2, "Building should be split into two parts")
        
        building_a = self.mapper.buildings_with_parcel_info[self.mapper.buildings_with_parcel_info['parcel_id'] == 1].iloc[0]
        building_b = self.mapper.buildings_with_parcel_info[self.mapper.buildings_with_parcel_info['parcel_id'] == 2].iloc[0]
        parcel_a = self.mapper.df_parcels[self.mapper.df_parcels['parcel_id'] == 1].iloc[0]
        parcel_b = self.mapper.df_parcels[self.mapper.df_parcels['parcel_id'] == 2].iloc[0]
        
        # Ensure the building was correctly split
        self.assertNotEqual(building_a['geometry'], building_b['geometry'], "Split buildings should have different geometries")
        
        # Ensure Building A and Parcel A have the same color
        self.assertEqual(building_a['color'], parcel_a['color'], "Building A and Parcel A should have the same color")
        
        # Ensure Building B and Parcel B have the same color
        self.assertEqual(building_b['color'], parcel_b['color'], "Building B and Parcel B should have the same color")
        
        # Ensure colors of A and B are different
        self.assertNotEqual(building_a['color'], building_b['color'], "Colors of Building A and Building B should be different")
        

    def test_assign_colors(self):
        # Ensure 'parcel_id' exists in buildings_with_parcel_info
        self.mapper.buildings_with_parcel_info['parcel_id'] = 1
        self.mapper.assign_colors()
        
        # Check if colors were assigned to parcels
        self.assertTrue('color' in self.mapper.df_parcels.columns)
        self.assertTrue(all(self.mapper.df_parcels['color'].str.startswith('#')))
        
        # Check if colors were assigned to buildings
        self.assertTrue('color' in self.mapper.buildings_with_parcel_info.columns)
        self.assertTrue(all(self.mapper.buildings_with_parcel_info['color'].str.startswith('#')))

    def test_subset(self):
        # Create more complex mock data for subset testing
        self.mapper.df_parcels = gpd.GeoDataFrame({
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                         Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
                         Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])],
            'parcel_id': [1, 2, 3]
        }, crs="EPSG:3857")
        self.mapper.buildings_with_parcel_info = gpd.GeoDataFrame({
            'geometry': [Polygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]),
                         Polygon([(1.2, 1.2), (1.8, 1.2), (1.8, 1.8), (1.2, 1.8)])],
            'building_id': [1, 2],
            'parcel_id': [1, 2]
        }, crs="EPSG:3857")
        
        subset_parcels, subset_buildings, bounds = self.mapper.subset(1, distance=1.5)
        
        self.assertEqual(len(subset_parcels), 3)  # Should include all parcels due to increased distance
        self.assertEqual(len(subset_buildings), 2)  # Should include both buildings
        self.assertIsInstance(bounds, tuple)
        self.assertEqual(len(bounds), 4)  # minx, miny, maxx, maxy

    @patch('PIL.Image.open')
    @patch('PIL.Image.new')
    def test_combine_images(self, mock_new, mock_open):
        # Create mock images
        mock_building_img = MagicMock()
        mock_building_img.width = 100
        mock_building_img.height = 200

        mock_parcel_img = MagicMock()
        mock_parcel_img.width = 100
        mock_parcel_img.height = 150

        # Set up mock Image.open to return our mock images
        mock_open.side_effect = [mock_building_img, mock_parcel_img]

        # Set up mock Image.new to return a new mock image
        mock_combined_img = MagicMock()
        mock_new.return_value = mock_combined_img

        # Call the method
        self.mapper.combine_images('mock_building.jpg', 'mock_parcel.jpg', 'mock_output.jpg')

        # Assert Image.open was called twice with correct arguments
        mock_open.assert_any_call('mock_building.jpg')
        mock_open.assert_any_call('mock_parcel.jpg')

        # Assert Image.new was called with correct arguments
        mock_new.assert_called_once_with('RGB', (200, 200))

        # Assert paste was called twice with correct arguments
        mock_combined_img.paste.assert_any_call(mock_building_img, (0, 0))
        mock_combined_img.paste.assert_any_call(mock_parcel_img, (100, 0))

        # Assert save was called with correct argument
        mock_combined_img.save.assert_called_once_with('mock_output.jpg')
if __name__ == '__main__':
    unittest.main()