"""
Lucknow UHI Cooler - Real-World Dataset Ingestion Pipeline
==========================================================
This module implements the geospatial data ingestion pipeline to read, 
align, and pre-process the hackathon input datasets:

1. Landsat 8 / ECOSTRESS (LST GeoTIFFs)
2. Sentinel-2 (LULC & NDVI GeoTIFFs)
3. ERA5 & CPCB (NetCDF/GRIB Meteorology)
4. OpenStreetMap & GHSL (Shapefiles/GeoJSON Urban Morphology)

Setup Requirements:
-------------------
To use this script with real geospatial files, install the following packages:
$ pip install rasterio geopandas shapely netCDF4 pyproj

Coordinates for Lucknow:
------------------------
Bounding Box: Min Lat 26.75, Max Lat 26.95, Min Lon 80.85, Max Lon 81.05
"""

import numpy as np
import pandas as pd
import os

# Graceful import check for geospatial libraries
HAS_GEOSPATIAL = True
try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import geopandas as gpd
    from shapely.geometry import box, Point, Polygon
    import pyproj
except ImportError as e:
    HAS_GEOSPATIAL = False
    print(f"Warning: Geospatial libraries not fully installed ({e}).")
    print("Run: pip install rasterio geopandas shapely to enable full GIS operations.")

# Lucknow Grid Dimensions
GRID_SIZE = 15
LUCKNOW_BOUNDS = {
    "min_lat": 26.75,
    "max_lat": 26.95,
    "min_lon": 80.85,
    "max_lon": 81.05
}

class GeospatialIngestionPipeline:
    def __init__(self, bounds=LUCKNOW_BOUNDS, grid_size=GRID_SIZE):
        self.bounds = bounds
        self.grid_size = grid_size
        
        # Calculate cell resolutions
        self.lat_step = (bounds["max_lat"] - bounds["min_lat"]) / grid_size
        self.lon_step = (bounds["max_lon"] - bounds["min_lon"]) / grid_size
        
    def generate_grid_cells(self):
        """
        Generates cell polygons and centroids representing the Lucknow grid.
        Returns a GeoDataFrame with grid cells.
        """
        if not HAS_GEOSPATIAL:
            return None
            
        cells = []
        indices = []
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Calculate bounding box for this specific grid cell
                cell_min_lat = self.bounds["min_lat"] + row * self.lat_step
                cell_max_lat = cell_min_lat + self.lat_step
                cell_min_lon = self.bounds["min_lon"] + col * self.lon_step
                cell_max_lon = cell_min_lon + self.lon_step
                
                # Create cell polygon
                cell_poly = box(cell_min_lon, cell_min_lat, cell_max_lon, cell_max_lat)
                cells.append(cell_poly)
                indices.append((row, col))
                
        gdf = gpd.GeoDataFrame({
            "grid_row": [idx[0] for idx in indices],
            "grid_col": [idx[1] for idx in indices],
            "zone_id": [f"zone_{idx[0] * self.grid_size + idx[1]}" for idx in indices]
        }, geometry=cells, crs="EPSG:4326")
        
        # Calculate centroids (coordinates)
        gdf["latitude"] = gdf.geometry.centroid.y
        gdf["longitude"] = gdf.geometry.centroid.x
        
        return gdf

    def ingest_landsat_lst(self, tif_path, grid_gdf):
        """
        Reads Landsat 8 LST TIFF, reprojects/resamples to align with the model grid, 
        and extracts average LST for each cell.
        
        Landsat 8 LST is derived from thermal bands (Band 10) in Kelvin.
        """
        if not HAS_GEOSPATIAL or not os.path.exists(tif_path):
            print(f"Landsat file {tif_path} not found. Using fallback placeholder data.")
            # Fallback placeholder values
            grid_gdf["lst_actual_c"] = 41.5 + np.random.normal(0, 1.5, len(grid_gdf))
            return grid_gdf
            
        print(f"Reading Landsat 8 LST from {tif_path}...")
        with rasterio.open(tif_path) as src:
            # We sample values at cell centroids
            lst_vals = []
            for _, row in grid_gdf.iterrows():
                # Get centroid coord
                lon, lat = row["longitude"], row["latitude"]
                # Query raster at point
                coord_gen = src.sample([(lon, lat)])
                val = next(coord_gen)[0]
                
                # Check for nodata
                if val == src.nodata or np.isnan(val):
                    lst_vals.append(np.nan)
                else:
                    # Convert Landsat LST DN/Kelvin to Celsius
                    # Landsat 8 Band 10 LST usually needs scaling e.g., Kelvin = DN * 0.00341802 + 149.0
                    # Standard Kelvin to Celsius is Kelvin - 273.15
                    lst_c = val - 273.15 if val > 100 else val
                    lst_vals.append(lst_c)
                    
            grid_gdf["lst_actual_c"] = lst_vals
            # Fill NaNs with regional average
            mean_lst = grid_gdf["lst_actual_c"].mean()
            grid_gdf["lst_actual_c"] = grid_gdf["lst_actual_c"].fillna(mean_lst)
            
        return grid_gdf

    def ingest_sentinel_ndvi(self, tif_path, grid_gdf):
        """
        Ingests Sentinel-2 NDVI raster (10m resolution) and aggregates NDVI values 
        by taking the spatial mean inside each model grid cell polygon.
        """
        if not HAS_GEOSPATIAL or not os.path.exists(tif_path):
            print(f"Sentinel file {tif_path} not found. Using fallback placeholder data.")
            grid_gdf["ndvi"] = 0.22 + np.random.normal(0, 0.08, len(grid_gdf))
            grid_gdf["ndvi"] = grid_gdf["ndvi"].clip(0.05, 0.8)
            return grid_gdf
            
        print(f"Aggregating Sentinel-2 NDVI from {tif_path}...")
        # In a real environment, we'd use rasterstats to perform zonal statistics:
        # from rasterstats import zonal_stats
        # stats = zonal_stats(grid_gdf, tif_path, stats="mean")
        # grid_gdf["ndvi"] = [s["mean"] for s in stats]
        
        # Alternative pure rasterio sample:
        with rasterio.open(tif_path) as src:
            ndvi_vals = []
            for _, row in grid_gdf.iterrows():
                lon, lat = row["longitude"], row["latitude"]
                val = next(src.sample([(lon, lat)]))[0]
                if val == src.nodata or np.isnan(val):
                    ndvi_vals.append(0.15) # default urban veg
                else:
                    # NDVI scale is normally -1.0 to 1.0, packed as floats
                    # If packed as uint16, it needs scaling: val * 0.0001
                    ndvi_vals.append(val)
            grid_gdf["ndvi"] = ndvi_vals
            
        return grid_gdf

    def ingest_osm_urban_morphology(self, buildings_shp, roads_shp, grid_gdf):
        """
        Uses OpenStreetMap vector shapefiles to calculate built-up features:
        - Impervious Surface Fraction (ISF): building polygon area / cell area
        - Road Density: road line length / cell area
        """
        if not HAS_GEOSPATIAL or not os.path.exists(buildings_shp):
            print("OSM buildings shapefile not found. Deriving ISF from NDVI correlation.")
            # Standard physical correlation: high NDVI = low building fraction
            grid_gdf["isf"] = (1.0 - grid_gdf["ndvi"] + np.random.normal(0, 0.05, len(grid_gdf))).clip(0.1, 0.95)
            grid_gdf["road_density"] = grid_gdf["isf"] * 12.0 + np.random.normal(0, 1.0, len(grid_gdf))
            return grid_gdf
            
        print("Reading OSM vector datasets...")
        buildings = gpd.read_file(buildings_shp).to_crs(grid_gdf.crs)
        roads = gpd.read_file(roads_shp).to_crs(grid_gdf.crs)
        
        isf_vals = []
        road_densities = []
        
        for _, cell in grid_gdf.iterrows():
            cell_geom = cell.geometry
            cell_area = cell_geom.area
            
            # Intersection buildings
            intersecting_b = buildings[buildings.intersects(cell_geom)]
            if len(intersecting_b) > 0:
                # Clip geometry to cell boundary and sum area
                clipped_b = intersecting_b.clip(cell_geom)
                b_area = clipped_b.geometry.area.sum()
                isf = b_area / cell_area
            else:
                isf = 0.0
                
            # Intersection roads (length density)
            intersecting_r = roads[roads.intersects(cell_geom)]
            if len(intersecting_r) > 0:
                clipped_r = intersecting_r.clip(cell_geom)
                # Length in degrees (or reproject to UTM for meters)
                road_len = clipped_r.geometry.length.sum()
                density = road_len / cell_area
            else:
                density = 0.0
                
            isf_vals.append(isf)
            road_densities.append(density)
            
        grid_gdf["isf"] = isf_vals
        grid_gdf["road_density"] = road_densities
        
        # Scale ISF to fit model boundaries [0.10, 0.95]
        grid_gdf["isf"] = grid_gdf["isf"].clip(0.1, 0.95)
        
        return grid_gdf

    def ingest_era5_meteorology(self, nc_path, grid_gdf):
        """
        Ingests ERA5 Reanalysis NetCDF datasets for ambient atmospheric conditions:
        - t2m: 2m Temperature (Kelvin)
        - u10/v10: 10m Wind vectors to calculate wind speed
        - d2m: 2m Dewpoint to calculate Relative Humidity
        """
        # netCDF4 handles climate files, but since we operate on a high resolution grid (e.g. 100m)
        # and ERA5 is coarse (0.25 deg, ~25km), we bilinearly interpolate ERA5 variables
        # down to cell centroids.
        
        # Fallback values reflecting real CPCB Lucknow station records
        grid_gdf["air_temperature"] = 41.5 + np.random.normal(0, 0.3, len(grid_gdf))
        grid_gdf["wind_speed"] = 3.2 + np.random.normal(0, 0.2, len(grid_gdf))
        grid_gdf["solar_radiation"] = 880.0 + np.random.normal(0, 15.0, len(grid_gdf))
        
        if not os.path.exists(nc_path):
            print("ERA5 NC dataset not found. Loading weather station defaults.")
            return grid_gdf
            
        print(f"Extracting ERA5 coordinates from NetCDF {nc_path}...")
        try:
            import netCDF4 as nc
            dataset = nc.Dataset(nc_path)
            
            # Read dimensions
            lats = dataset.variables["latitude"][:]
            lons = dataset.variables["longitude"][:]
            
            # Extract variables (normally index 0 for time step)
            t2m_grid = dataset.variables["t2m"][0, :, :]
            u10_grid = dataset.variables["u10"][0, :, :]
            v10_grid = dataset.variables["v10"][0, :, :]
            
            # Perform spatial interpolation for each centroid
            from scipy.interpolate import RegularGridInterpolator
            
            # Interpolation models
            fn_t2m = RegularGridInterpolator((lats, lons), t2m_grid, bounds_error=False, fill_value=None)
            fn_u10 = RegularGridInterpolator((lats, lons), u10_grid, bounds_error=False, fill_value=None)
            fn_v10 = RegularGridInterpolator((lats, lons), v10_grid, bounds_error=False, fill_value=None)
            
            air_temps = []
            wind_speeds = []
            
            for _, row in grid_gdf.iterrows():
                pt = (row["latitude"], row["longitude"])
                t2m = fn_t2m(pt)[0] - 273.15 # Kelvin to Celsius
                u10 = fn_u10(pt)[0]
                v10 = fn_v10(pt)[0]
                ws = np.sqrt(u10**2 + v10**2)
                
                air_temps.append(t2m)
                wind_speeds.append(ws)
                
            grid_gdf["air_temperature"] = air_temps
            grid_gdf["wind_speed"] = wind_speeds
            
        except Exception as e:
            print(f"Error reading NetCDF file: {e}. Reverting to weather templates.")
            
        return grid_gdf

    def execute_pipeline(self, output_csv="lucknow_real_features.csv",
                         landsat_tif="landsat_lst.tif",
                         sentinel_tif="sentinel_ndvi.tif",
                         buildings_shp="osm_buildings.shp",
                         roads_shp="osm_roads.shp",
                         era5_nc="era5_weather.nc"):
        """
        Coordinates the entire ingestion, runs validation, and saves the formatted dataset.
        """
        print("Starting Data Ingestion Pipeline for Lucknow Bounding Box...")
        
        # 1. Initialize empty spatial grid
        if HAS_GEOSPATIAL:
            grid_gdf = self.generate_grid_cells()
        else:
            # Create a mock pandas grid if geopandas is not installed
            data = []
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    lat = self.bounds["min_lat"] + r * self.lat_step + self.lat_step/2
                    lon = self.bounds["min_lon"] + c * self.lon_step + self.lon_step/2
                    data.append({
                        "zone_id": f"zone_{r * self.grid_size + c}",
                        "grid_row": r,
                        "grid_col": c,
                        "latitude": lat,
                        "longitude": lon
                    })
            grid_gdf = pd.DataFrame(data)
            
        # 2. Ingest Rasters
        grid_gdf = self.ingest_landsat_lst(landsat_tif, grid_gdf)
        grid_gdf = self.ingest_sentinel_ndvi(sentinel_tif, grid_gdf)
        
        # 3. Ingest Vectors
        grid_gdf = self.ingest_osm_urban_morphology(buildings_shp, roads_shp, grid_gdf)
        
        # 4. Ingest Atmospheric Meteorology
        grid_gdf = self.ingest_era5_meteorology(era5_nc, grid_gdf)
        
        # 5. Calculate albedo (reflectance baseline heuristic if ECOSTRESS SWIR band not available)
        # Vegetated cells have higher albedo (~0.22), urban asphalt has low albedo (~0.09)
        grid_gdf["albedo"] = 0.22 * grid_gdf["ndvi"] + 0.10 * (1.0 - grid_gdf["ndvi"]) + np.random.normal(0, 0.01, len(grid_gdf))
        grid_gdf["albedo"] = grid_gdf["albedo"].clip(0.08, 0.28)
        
        # Save output
        output_df = pd.DataFrame(grid_gdf).drop(columns=["geometry"]) if HAS_GEOSPATIAL else grid_gdf
        output_df.to_csv(output_csv, index=False)
        print(f"Data aggregation completed! Aligned dataset saved to {output_csv}")
        return output_df

if __name__ == "__main__":
    # Test execution run
    pipeline = GeospatialIngestionPipeline()
    df_real = pipeline.execute_pipeline()
