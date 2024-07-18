from building2parcel_trainingdata import ParcelBuildingMapper
import os

def test_parcel_building_mapper():
    
    parcels_path = parcels_path = "C:/Million Neighborhoods/Spatial Data/ny-manhattan-parcels/NYC_2021_Tax_Parcels_SHP_2203/NewYork_2021_Tax_Parcels_SHP_2203.shp"
    buildings_path = "C:/Million Neighborhoods/Spatial Data/ny-manhattan-buildings/geo_export_a80ea1a2-e8e0-4ffd-862c-1199433ac303.shp"
    
    
    mapper = ParcelBuildingMapper(parcels_path, buildings_path)
    
    
    output_path = "test_output/"
    os.makedirs(output_path, exist_ok=True)
    
    
    mapper.generate_maps(output_path, output_path, start_index=0, end_index=1, distance=100, map_type='simple')
    
    
    expected_files = [
        os.path.join(output_path, "parcels_0.jpg"),
        os.path.join(output_path, "buildings_0.jpg")
    ]
    
    for file in expected_files:
        assert os.path.exists(file), f"Expected output file {file} not found."
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_parcel_building_mapper()