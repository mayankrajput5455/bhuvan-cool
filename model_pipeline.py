import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

GRID_SIZE = 15

# Landmark configurations for each city to guide synthetic data generation
CITY_CONFIGS = {
    "lucknow": {
        "lat": 26.8467, "lon": 80.9462,
        "lat_range": 0.12, "lon_range": 0.14,
        "base_temp": 41.5, "base_wind": 3.2, "base_solar": 880.0,
        "water_type": "river_gomti",
        "landmarks": {
            "Chowk (Old City)": {"coord": (2.0, 11.0), "ndvi": 0.06, "pop_density": 42000.0, "income": 1, "elderly": 0.18, "albedo": 0.09, "desc": "Extremely dense historic area with narrow lanes, old brick/concrete, and minimal green cover."},
            "Hazratganj (CBD)": {"coord": (7.0, 7.0), "ndvi": 0.12, "pop_density": 26000.0, "income": 3, "elderly": 0.14, "albedo": 0.13, "desc": "Central business district, high commercial activity, paved surfaces, high vehicle heat."},
            "Gomti Nagar (Planned)": {"coord": (11.0, 6.0), "ndvi": 0.38, "pop_density": 9500.0, "income": 3, "elderly": 0.11, "albedo": 0.18, "desc": "Modern planned residential area near the river with wide tree-lined avenues and parks."},
            "Kukrail Forest Reserve": {"coord": (12.0, 13.0), "ndvi": 0.72, "pop_density": 800.0, "income": 2, "elderly": 0.08, "albedo": 0.22, "desc": "Large dense urban forest reserve providing regional cooling."},
            "Aminabad (Market)": {"coord": (5.0, 6.0), "ndvi": 0.07, "pop_density": 39000.0, "income": 2, "elderly": 0.16, "albedo": 0.10, "desc": "Dense commercial retail market with very high daily footfall and compact construction."},
            "Charbagh (Station)": {"coord": (4.0, 4.0), "ndvi": 0.09, "pop_density": 21000.0, "income": 2, "elderly": 0.12, "albedo": 0.11, "desc": "Major railway junction and transport hub, high asphalt coverage and heavy traffic."}
        }
    },
    "delhi": {
        "lat": 28.6139, "lon": 77.2090,
        "lat_range": 0.14, "lon_range": 0.16,
        "base_temp": 42.8, "base_wind": 2.8, "base_solar": 910.0,
        "water_type": "river_yamuna",
        "landmarks": {
            "Chandni Chowk": {"coord": (4.0, 11.0), "ndvi": 0.05, "pop_density": 46000.0, "income": 1, "elderly": 0.17, "albedo": 0.08, "desc": "Extremely dense historic bazaar in Old Delhi, heavily paved with immense thermal mass."},
            "Connaught Place": {"coord": (6.0, 7.0), "ndvi": 0.14, "pop_density": 24000.0, "income": 3, "elderly": 0.13, "albedo": 0.12, "desc": "Radial commercial center, heavy concrete facades, asphalt roads, and high AC load."},
            "Delhi Ridge Forest": {"coord": (4.0, 5.0), "ndvi": 0.65, "pop_density": 900.0, "income": 2, "elderly": 0.08, "albedo": 0.20, "desc": "Hilly forest reserve acting as green lungs for Delhi, yielding strong evapotranspiration cooling."},
            "Okhla Industrial Area": {"coord": (11.0, 3.0), "ndvi": 0.09, "pop_density": 28000.0, "income": 2, "elderly": 0.10, "albedo": 0.11, "desc": "Dense manufacturing area with large metal/asphalt rooftops, high thermal absorption."},
            "Dwarka Sub-city": {"coord": (2.0, 4.0), "ndvi": 0.24, "pop_density": 16000.0, "income": 3, "elderly": 0.12, "albedo": 0.17, "desc": "Planned residential suburb, moderate street lining tree cover and parks."}
        }
    },
    "kanpur": {
        "lat": 26.4499, "lon": 80.3319,
        "lat_range": 0.12, "lon_range": 0.14,
        "base_temp": 42.0, "base_wind": 3.0, "base_solar": 890.0,
        "water_type": "river_ganges",
        "landmarks": {
            "Jajmau Industrial": {"coord": (12.0, 4.0), "ndvi": 0.07, "pop_density": 32000.0, "income": 1, "elderly": 0.15, "albedo": 0.09, "desc": "Major industrial cluster on the Ganges banks, high metal roof fractions and road density."},
            "Civil Lines (Kanpur)": {"coord": (7.0, 8.0), "ndvi": 0.12, "pop_density": 22000.0, "income": 3, "elderly": 0.14, "albedo": 0.13, "desc": "Central business and historic residential sector, high concrete building layout."},
            "Kalyanpur (Suburbs)": {"coord": (3.0, 11.0), "ndvi": 0.23, "pop_density": 14000.0, "income": 2, "elderly": 0.11, "albedo": 0.16, "desc": "Residential development sector, moderate open spaces and gardens."},
            "IIT Kanpur Campus": {"coord": (2.0, 13.0), "ndvi": 0.52, "pop_density": 3500.0, "income": 3, "elderly": 0.08, "albedo": 0.19, "desc": "Vastly vegetated institutional campus, dense canopy tree lines creating cool air micro-buffers."}
        }
    },
    "goa": {
        "lat": 15.4909, "lon": 73.8278,
        "lat_range": 0.10, "lon_range": 0.12,
        "base_temp": 33.5, "base_wind": 4.5, "base_solar": 820.0,
        "water_type": "coastline_goa",
        "landmarks": {
            "Panaji Centro": {"coord": (5.0, 8.0), "ndvi": 0.24, "pop_density": 12000.0, "income": 3, "elderly": 0.16, "albedo": 0.15, "desc": "Capital town, medium density, typical colonial buildings with terracotta tiled roofs."},
            "Miramar Beach Coast": {"coord": (2.0, 5.0), "ndvi": 0.08, "pop_density": 1500.0, "income": 3, "elderly": 0.11, "albedo": 0.38, "desc": "Sandy shorefront with very high solar albedo and direct coastal wind exposure."},
            "Saligao Canopy Forest": {"coord": (9.0, 12.0), "ndvi": 0.76, "pop_density": 600.0, "income": 2, "elderly": 0.10, "albedo": 0.21, "desc": "Dense tropical evergreen forest canopy typical of inland Goan foothills."},
            "Marmagao Vasco Port": {"coord": (3.0, 2.0), "ndvi": 0.06, "pop_density": 18000.0, "income": 2, "elderly": 0.13, "albedo": 0.10, "desc": "Industrial port infrastructure, high asphalt fraction and shipping container heat loads."}
        }
    },
    "mumbai": {
        "lat": 19.0760, "lon": 72.8777,
        "lat_range": 0.20, "lon_range": 0.12,
        "base_temp": 34.2, "base_wind": 4.2, "base_solar": 840.0,
        "water_type": "coastline_mumbai",
        "landmarks": {
            "Dharavi (Ultra-Dense)": {"coord": (7.0, 7.0), "ndvi": 0.04, "pop_density": 52000.0, "income": 1, "elderly": 0.10, "albedo": 0.09, "desc": "One of the densest residential blocks worldwide, metal sheeting roofs, zero vegetation and extreme heat retention."},
            "Nariman Point (CBD)": {"coord": (4.0, 1.0), "ndvi": 0.08, "pop_density": 18000.0, "income": 3, "elderly": 0.15, "albedo": 0.12, "desc": "Financial skyline, high-rise concrete canyons, heavy vehicular load, and maritime air interface."},
            "Sanjay Gandhi Park": {"coord": (9.0, 12.0), "ndvi": 0.80, "pop_density": 500.0, "income": 2, "elderly": 0.07, "albedo": 0.22, "desc": "Massive protected national forest inside the city bounds, yielding intense local cooling."},
            "Bandra Kurla (BKC)": {"coord": (7.0, 6.0), "ndvi": 0.15, "pop_density": 22000.0, "income": 3, "elderly": 0.11, "albedo": 0.16, "desc": "Modern commercial center, wide paved streets, glass buildings, and intense microclimate trapping."}
        }
    }
}

def get_distance_to_water_body(city, x, y):
    """
    Computes distance to river centerline or coastline based on city characteristics.
    """
    cfg = CITY_CONFIGS[city]
    wtype = cfg["water_type"]
    
    if wtype == "river_gomti":
        # Gomti River centerline
        river_points = [(13.0 - 0.8 * yr - 2.0 * np.sin(yr / 2.0), yr) for yr in np.linspace(0, GRID_SIZE-1, 50)]
        return min(np.sqrt((x - xr)**2 + (y - yr)**2) for xr, yr in river_points)
        
    elif wtype == "river_yamuna":
        # Yamuna River (NW to SE in Delhi)
        river_points = [(9.0 - 0.3 * yr - 1.5 * np.cos(yr / 3.0), yr) for yr in np.linspace(0, GRID_SIZE-1, 50)]
        return min(np.sqrt((x - xr)**2 + (y - yr)**2) for xr, yr in river_points)
        
    elif wtype == "river_ganges":
        # Ganges River along Kanpur north boundary (y approx 13-14)
        river_points = [(xr, 13.0 + 0.5 * np.sin(xr / 2.0)) for xr in np.linspace(0, GRID_SIZE-1, 50)]
        return min(np.sqrt((x - xr)**2 + (y - yr)**2) for xr, yr in river_points)
        
    elif wtype == "coastline_goa":
        # Goa Coast: West side is sea (x <= 3)
        if x <= 3.0:
            return 0.0
        return float(x - 3.0)
        
    elif wtype == "coastline_mumbai":
        # Mumbai Peninsula: West (x <= 3) and East (x >= 12) is sea/creek
        if x <= 3.0:
            return 0.0
        elif x >= 12.0:
            return 0.0
        return float(min(x - 3.0, 12.0 - x))
        
    return 999.0

def calculate_physics_lst(air_temp, solar_rad, albedo, wind_speed, ndvi):
    h_c = 5.8 + 4.1 * wind_speed
    rad_forcing = ((1.0 - albedo) * solar_rad) / h_c
    evap_cooling = 9.0 * ndvi * (solar_rad / 900.0)
    return air_temp + rad_forcing - evap_cooling

def generate_city_dataset(city_name):
    """
    Generates synthetic grid dataset for a given city based on its unique config.
    """
    np.random.seed(42 if city_name == "lucknow" else ord(city_name[0]))
    cfg = CITY_CONFIGS[city_name]
    landmarks = cfg["landmarks"]
    
    data = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x = float(col)
            y = float(row)
            
            # Map grid to coordinates
            lat = cfg["lat"] - cfg["lat_range"]/2 + (row / (GRID_SIZE - 1)) * cfg["lat_range"]
            lon = cfg["lon"] - cfg["lon_range"]/2 + (col / (GRID_SIZE - 1)) * cfg["lon_range"]
            zone_id = f"zone_{row * GRID_SIZE + col}"
            
            d_water = get_distance_to_water_body(city_name, x, y)
            
            # Identify water cell flag
            is_water = False
            if city_name in ["goa", "mumbai"] and d_water == 0.0:
                is_water = True
                
            # Interp attributes from landmarks
            total_weight = 0.0
            weighted_ndvi = 0.0
            weighted_pop = 0.0
            weighted_income = 0.0
            weighted_elderly = 0.0
            weighted_albedo = 0.0
            
            closest_landmark_name = "Zone"
            min_dist = 999.0
            
            for name, landmark in landmarks.items():
                lx, ly = landmark["coord"]
                dist = np.sqrt((x - lx)**2 + (y - ly)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_landmark_name = name
                    
                w = 1.0 / (dist**2 + 0.5)
                total_weight += w
                weighted_ndvi += landmark["ndvi"] * w
                weighted_pop += landmark["pop_density"] * w
                weighted_income += landmark["income"] * w
                weighted_elderly += landmark["elderly"] * w
                weighted_albedo += landmark["albedo"] * w
                
            ndvi = weighted_ndvi / total_weight
            pop_density = weighted_pop / total_weight
            income_level = int(round(weighted_income / total_weight))
            elderly_ratio = weighted_elderly / total_weight
            albedo = weighted_albedo / total_weight
            
            # Environmental cooling / humidity scaling near water bodies
            water_cool_factor = np.exp(-0.4 * d_water)
            
            if is_water:
                ndvi = 0.02
                albedo = 0.05
                isf = 0.0
                pop_density = 0.0
                local_wind_speed = cfg["base_wind"] * 1.3
                local_solar_rad = cfg["base_solar"]
                local_air_temp = cfg["base_temp"] - 3.5
                lst_physics = local_air_temp + 0.5
                lst_day = lst_physics
                lst_night = cfg["base_temp"] - 4.5
                name_str = "Water Body / Sea Area"
            else:
                # Apply water buffer influence
                ndvi = np.clip(ndvi + 0.12 * water_cool_factor + np.random.normal(0, 0.02), 0.05, 0.85)
                albedo = np.clip(albedo - 0.04 * water_cool_factor + np.random.normal(0, 0.01), 0.05, 0.6)
                
                isf = np.clip(1.0 - ndvi - 0.15 * water_cool_factor + np.random.normal(0, 0.04), 0.1, 0.95)
                pop_density = np.clip(pop_density + np.random.normal(0, 1500), 500, 48000)
                
                local_wind_speed = np.clip(
                    cfg["base_wind"] * (1.0 - 0.35 * isf) + 0.8 * water_cool_factor + np.random.normal(0, 0.15),
                    0.8, 6.0
                )
                local_solar_rad = np.clip(
                    cfg["base_solar"] * (1.0 - 0.05 * isf) + np.random.normal(0, 10.0),
                    700.0, 950.0
                )
                local_air_temp = cfg["base_temp"] + 1.2 * isf - 0.8 * water_cool_factor + np.random.normal(0, 0.1)
                
                # Baseline LST Physics
                lst_physics = calculate_physics_lst(local_air_temp, local_solar_rad, albedo, local_wind_speed, ndvi)
                
                # Actual LST (XGBoost target includes complex residual)
                residual = (
                    6.2 * (isf ** 2.2) + 
                    3.0 * (pop_density / 30000.0) - 
                    2.5 * water_cool_factor + 
                    np.random.normal(0.0, 0.45)
                )
                lst_day = lst_physics + residual
                
                # Night LST (Urban thermal inertia release)
                regional_air_temp_night = cfg["base_temp"] - 12.0
                lst_night = (
                    regional_air_temp_night + 
                    4.5 * isf - 
                    2.8 * ndvi - 
                    1.1 * local_wind_speed + 
                    1.8 * (pop_density / 30000.0) - 
                    1.5 * water_cool_factor + 
                    np.random.normal(0, 0.35)
                )
                
                if min_dist < 1.5:
                    sector = int((x + y) % 3) + 1
                    name_str = f"{closest_landmark_name} Sec {sector}"
                else:
                    name_str = f"Zone ({int(x)},{int(y)})"
                    
            data.append({
                "city": city_name,
                "zone_id": zone_id,
                "x": int(x),
                "y": int(y),
                "name": name_str,
                "latitude": lat,
                "longitude": lon,
                "ndvi": float(ndvi),
                "albedo": float(albedo),
                "isf": float(isf),
                "pop_density": float(pop_density),
                "income_level": int(income_level),
                "elderly_ratio": float(elderly_ratio),
                "wind_speed": float(local_wind_speed),
                "wind_direction": 240.0,
                "solar_radiation": float(local_solar_rad),
                "air_temperature": float(local_air_temp),
                "distance_to_river": float(d_water),  # Map distance field universally
                "is_water": bool(is_water),
                "lst_physics": float(lst_physics),
                "lst_day_actual": float(lst_day),
                "lst_night_actual": float(lst_night)
            })
            
    return pd.DataFrame(data)

def train_physics_informed_model(df):
    features = [
        "ndvi", "albedo", "isf", "pop_density", 
        "wind_speed", "solar_radiation", "air_temperature"
    ]
    df["residual"] = df["lst_day_actual"] - df["lst_physics"]
    
    # Exclude open water bodies from residual fitting since they fit purely physically
    train_df = df[df["is_water"] == False]
    
    X = train_df[features]
    y = train_df["residual"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    
    print(f"Unified Multi-City XGBoost Model Trained.")
    print(f"Combined Test Set MAE: {mae:.4f} °C, RMSE: {rmse:.4f} °C")
    
    with open("xgb_residual_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    try:
        explainer = shap.TreeExplainer(model)
        with open("shap_explainer.pkl", "wb") as f:
            pickle.dump(explainer, f)
    except Exception as e:
        print(f"Warning: shap.TreeExplainer initialization failed: {e}")
        explainer = None
        with open("shap_explainer.pkl", "wb") as f:
            pickle.dump(None, f)
            
    return model, explainer, df

def calculate_hvi_and_clusters(df):
    scaler = MinMaxScaler()
    
    # Exclude water zones from HVI rankings (water zones are resilient/neutral)
    land_mask = df["is_water"] == False
    
    df["hvi"] = 0.0
    df["hvi_class"] = "Water Body"
    
    if land_mask.any():
        land_df = df[land_mask].copy()
        
        df_indicators = pd.DataFrame({
            "lst_day_norm": scaler.fit_transform(land_df[["lst_day_actual"]]).flatten(),
            "lst_night_norm": scaler.fit_transform(land_df[["lst_night_actual"]]).flatten(),
            "pop_density_norm": scaler.fit_transform(land_df[["pop_density"]]).flatten(),
            "elderly_norm": scaler.fit_transform(land_df[["elderly_ratio"]]).flatten(),
            "income_norm": 1.0 - (land_df["income_level"] - 1.0) / 2.0
        })
        
        hvi_scores = (
            0.20 * df_indicators["lst_day_norm"] +
            0.20 * df_indicators["lst_night_norm"] +
            0.15 * df_indicators["pop_density_norm"] +
            0.15 * df_indicators["elderly_norm"] +
            0.30 * df_indicators["income_norm"]
        )
        
        df.loc[land_mask, "hvi"] = hvi_scores
        
        def classify_hvi(score):
            if score < 0.35: return "Low Risk"
            elif score < 0.52: return "Medium Risk"
            elif score < 0.68: return "High Risk"
            else: return "Extreme Risk"
            
        df.loc[land_mask, "hvi_class"] = df.loc[land_mask, "hvi"].apply(classify_hvi)
        
        # Priority Zone Clustering
        X_clust = land_df[["lst_day_actual", "hvi"]].copy()
        X_clust_norm = scaler.fit_transform(X_clust)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_clust_norm)
        
        cluster_centers = kmeans.cluster_centers_
        cluster_severity = np.sum(cluster_centers, axis=1)
        sorted_idx = np.argsort(cluster_severity)
        mapping = {old: new for new, old in enumerate(sorted_idx)}
        
        df.loc[land_mask, "cluster"] = pd.Series(clusters, index=land_df.index).map(mapping)
    
    # Water zones defaults
    df.loc[~land_mask, "cluster"] = -1
    
    cluster_labels = {
        -1: "Open Water Body",
        0: "Cool & Resilient (Safe)",
        1: "Moderate Heat / Moderate Risk",
        2: "High Heat / Medium Risk (Heat-Dominant)",
        3: "High Heat & High HVI (Critical Priority Zone)"
    }
    df["cluster_label"] = df["cluster"].map(cluster_labels)
    
    # SUHII Nighttime
    rural_zones = df[(df["isf"] < 0.25) & (df["pop_density"] < 3000) & (df["is_water"] == False)]
    if len(rural_zones) > 0:
        rural_baseline_day = rural_zones["lst_day_actual"].mean()
        rural_baseline_night = rural_zones["lst_night_actual"].mean()
    else:
        rural_baseline_day = df[df["is_water"] == False]["lst_day_actual"].nsmallest(15).mean()
        rural_baseline_night = df[df["is_water"] == False]["lst_night_actual"].nsmallest(15).mean()
        
    df["suhii_day"] = df["lst_day_actual"] - rural_baseline_day
    df["suhii_night"] = df["lst_night_actual"] - rural_baseline_night
    
    # Ventilation Suitability (water cells are excellent ventilation avenues!)
    df["ventilation_suitability"] = np.clip(
        (1.0 - df["isf"]) * 0.6 + (df["wind_speed"] / 6.0) * 0.4 + 0.2 * np.exp(-0.25 * df["distance_to_river"]),
        0.0, 1.0
    )
    df.loc[df["is_water"] == True, "ventilation_suitability"] = 1.0
    
    return df

def get_ventilation_corridor_paths(df):
    corridor_cells = []
    city = df["city"].iloc[0]
    wtype = CITY_CONFIGS[city]["water_type"]
    
    if wtype in ["river_gomti", "river_yamuna", "river_ganges"]:
        river_cells = df[df["distance_to_river"] < 1.25]["zone_id"].tolist()
        corridor_cells.append({
            "name": "Blue-Green Riverine Corridor",
            "zones": river_cells,
            "type": "Riverine Corridor"
        })
        
        forest_cells = df[(df["ndvi"] > 0.35) & (df["y"] > 8) & (df["x"] > 7)]["zone_id"].tolist()
        if forest_cells:
            corridor_cells.append({
                "name": "Green Canopy wind corridor",
                "zones": forest_cells,
                "type": "Vegetative Corridor"
            })
    else:
        # Coastal flow corridors for Mumbai and Goa
        coastal_zones = df[df["is_water"] == True]["zone_id"].tolist()
        corridor_cells.append({
            "name": "Maritime Cool Sea Breeze Channel",
            "zones": coastal_zones,
            "type": "Maritime Corridor"
        })
        
        green_breeze = df[(df["ndvi"] > 0.35) & (df["distance_to_river"] < 3.0) & (df["is_water"] == False)]["zone_id"].tolist()
        if green_breeze:
            corridor_cells.append({
                "name": "Coastal Vegetation Buffer",
                "zones": green_breeze,
                "type": "Ecotone Corridor"
            })
            
    return corridor_cells

def simulate_intervention(zone_data, intervention_type, intensity_pct):
    if zone_data.get("is_water", False):
        return {
            "ndvi": 0.02, "albedo": 0.05, "isf": 0.0, 
            "wind_speed": zone_data["wind_speed"], "lst_physics": zone_data["lst_physics"], "cost": 0.0
        }
        
    new_data = zone_data.copy()
    intensity = float(intensity_pct)
    
    cost = 0.0
    ndvi_delta = 0.0
    albedo_delta = 0.0
    isf_delta = 0.0
    
    if intervention_type == "tree_planting":
        cost = 80000.0 * intensity
        ndvi_delta = 0.008 * intensity
        isf_delta = -0.008 * intensity
    elif intervention_type == "green_roofs":
        cost = 150000.0 * intensity
        ndvi_delta = 0.005 * intensity
        albedo_delta = 0.003 * intensity
        isf_delta = -0.005 * intensity
    elif intervention_type == "cool_pavement":
        cost = 100000.0 * intensity
        albedo_delta = 0.006 * intensity
        
    new_data["ndvi"] = np.clip(new_data["ndvi"] + ndvi_delta, 0.05, 0.85)
    new_data["albedo"] = np.clip(new_data["albedo"] + albedo_delta, 0.05, 0.65)
    new_data["isf"] = np.clip(new_data["isf"] + isf_delta, 0.05, 0.95)
    
    cfg = CITY_CONFIGS[zone_data["city"]]
    water_cool_factor = np.exp(-0.4 * new_data["distance_to_river"])
    new_data["wind_speed"] = np.clip(
        cfg["base_wind"] * (1.0 - 0.35 * new_data["isf"]) + 0.8 * water_cool_factor,
        0.8, 6.0
    )
    
    new_lst_phys = calculate_physics_lst(
        new_data["air_temperature"], new_data["solar_radiation"],
        new_data["albedo"], new_data["wind_speed"], new_data["ndvi"]
    )
    
    return {
        "ndvi": float(new_data["ndvi"]),
        "albedo": float(new_data["albedo"]),
        "isf": float(new_data["isf"]),
        "wind_speed": float(new_data["wind_speed"]),
        "lst_physics": float(new_lst_phys),
        "cost": float(cost)
    }

def run_simulation_inference(df, model, zone_interventions):
    df_sim = df.copy()
    features = ["ndvi", "albedo", "isf", "pop_density", "wind_speed", "solar_radiation", "air_temperature"]
    total_cost = 0.0
    
    for zone_id, interventions in zone_interventions.items():
        idx = df_sim[df_sim["zone_id"] == zone_id].index
        if len(idx) == 0: continue
        idx = idx[0]
        
        # Don't apply changes on water cells
        if df_sim.loc[idx, "is_water"]:
            continue
            
        zone_data = df_sim.loc[idx].to_dict()
        accumulated_changes = {
            "city": zone_data["city"],
            "is_water": zone_data["is_water"],
            "ndvi": zone_data["ndvi"],
            "albedo": zone_data["albedo"],
            "isf": zone_data["isf"],
            "wind_speed": zone_data["wind_speed"],
            "air_temperature": zone_data["air_temperature"],
            "solar_radiation": zone_data["solar_radiation"],
            "distance_to_river": zone_data["distance_to_river"]
        }
        
        zone_cost = 0.0
        for itype, intensity in interventions.items():
            if intensity <= 0: continue
            sim_res = simulate_intervention(accumulated_changes, itype, intensity)
            accumulated_changes["ndvi"] = sim_res["ndvi"]
            accumulated_changes["albedo"] = sim_res["albedo"]
            accumulated_changes["isf"] = sim_res["isf"]
            accumulated_changes["wind_speed"] = sim_res["wind_speed"]
            accumulated_changes["lst_physics"] = sim_res["lst_physics"]
            zone_cost += sim_res["cost"]
            
        total_cost += zone_cost
        
        df_sim.loc[idx, "ndvi"] = accumulated_changes["ndvi"]
        df_sim.loc[idx, "albedo"] = accumulated_changes["albedo"]
        df_sim.loc[idx, "isf"] = accumulated_changes["isf"]
        df_sim.loc[idx, "wind_speed"] = accumulated_changes["wind_speed"]
        df_sim.loc[idx, "lst_physics"] = accumulated_changes["lst_physics"]
        
    # Re-run XGBoost residual predictor on land cells
    land_mask = df_sim["is_water"] == False
    if land_mask.any():
        X_sim = df_sim.loc[land_mask, features]
        df_sim.loc[land_mask, "residual_pred"] = model.predict(X_sim)
        df_sim.loc[land_mask, "lst_day_pred"] = df_sim.loc[land_mask, "lst_physics"] + df_sim.loc[land_mask, "residual_pred"]
    
    # Water cells remain same
    df_sim.loc[~land_mask, "lst_day_pred"] = df_sim.loc[~land_mask, "lst_physics"]
    df_sim["lst_delta"] = df["lst_day_actual"] - df_sim["lst_day_pred"]
    
    # Recalculate nighttime LST based on modified morphology
    for idx, row in df_sim.iterrows():
        if row["is_water"]: continue
        cfg = CITY_CONFIGS[row["city"]]
        regional_air_temp_night = cfg["base_temp"] - 12.0
        water_cool_factor = np.exp(-0.4 * row["distance_to_river"])
        df_sim.loc[idx, "lst_night_pred"] = (
            regional_air_temp_night + 
            4.5 * row["isf"] - 
            2.8 * row["ndvi"] - 
            1.1 * row["wind_speed"] + 
            1.8 * (row["pop_density"] / 30000.0) - 
            1.5 * water_cool_factor
        )
        
    return df_sim, total_cost

def run_greedy_optimizer(df, model, budget_inr, weight_by_hvi=True):
    df_current = df.copy()
    allocation = {row["zone_id"]: {"tree_planting": 0, "green_roofs": 0, "cool_pavement": 0} for _, row in df.iterrows()}
    
    remaining_budget = float(budget_inr)
    total_spent = 0.0
    step_pct = 5.0
    
    interventions_meta = {
        "tree_planting": {"unit_cost": 80000.0 * step_pct, "max_limit": 30.0},
        "green_roofs": {"unit_cost": 150000.0 * step_pct, "max_limit": 40.0},
        "cool_pavement": {"unit_cost": 100000.0 * step_pct, "max_limit": 40.0}
    }
    
    iteration = 0
    max_iterations = 800
    
    while remaining_budget > 400000.0 and iteration < max_iterations:
        best_roi = -1.0
        best_choice = None
        
        for _, row in df_current.iterrows():
            if row["is_water"]: continue
            zone_id = row["zone_id"]
            hvi_val = row["hvi"]
            
            current_tree = allocation[zone_id]["tree_planting"]
            current_green = allocation[zone_id]["green_roofs"]
            
            for itype, meta in interventions_meta.items():
                cost = meta["unit_cost"]
                if cost > remaining_budget: continue
                if allocation[zone_id][itype] + step_pct > meta["max_limit"]: continue
                if itype in ["tree_planting", "green_roofs"] and (current_tree + current_green + step_pct > 60.0):
                    continue
                
                zone_dict = row.to_dict()
                sim_meta = simulate_intervention(zone_dict, itype, step_pct)
                
                features_sim = pd.DataFrame([{
                    "ndvi": sim_meta["ndvi"], "albedo": sim_meta["albedo"], "isf": sim_meta["isf"],
                    "pop_density": zone_dict["pop_density"], "wind_speed": sim_meta["wind_speed"],
                    "solar_radiation": zone_dict["solar_radiation"], "air_temperature": zone_dict["air_temperature"]
                }])
                
                res_pred = model.predict(features_sim)[0]
                temp_pred_new = sim_meta["lst_physics"] + res_pred
                
                delta_t = row["lst_day_actual"] - temp_pred_new
                benefit = delta_t
                if weight_by_hvi:
                    benefit *= hvi_val
                    
                roi = benefit / cost
                if roi > best_roi and delta_t > 0.01:
                    best_roi = roi
                    best_choice = (zone_id, itype)
                    
        if best_choice is not None:
            zone_id, itype = best_choice
            cost = interventions_meta[itype]["unit_cost"]
            
            idx = df_current[df_current["zone_id"] == zone_id].index[0]
            zone_row = df_current.loc[idx].to_dict()
            
            sim_meta = simulate_intervention(zone_row, itype, step_pct)
            df_current.loc[idx, "ndvi"] = sim_meta["ndvi"]
            df_current.loc[idx, "albedo"] = sim_meta["albedo"]
            df_current.loc[idx, "isf"] = sim_meta["isf"]
            df_current.loc[idx, "wind_speed"] = sim_meta["wind_speed"]
            df_current.loc[idx, "lst_physics"] = sim_meta["lst_physics"]
            
            feat_df = df_current.loc[[idx], ["ndvi", "albedo", "isf", "pop_density", "wind_speed", "solar_radiation", "air_temperature"]]
            res_pred = model.predict(feat_df)[0]
            df_current.loc[idx, "lst_day_actual"] = sim_meta["lst_physics"] + res_pred
            
            allocation[zone_id][itype] += step_pct
            remaining_budget -= cost
            total_spent += cost
            iteration += 1
        else:
            break
            
    rec_list = []
    for zone_id, alloc in allocation.items():
        if alloc["tree_planting"] > 0 or alloc["green_roofs"] > 0 or alloc["cool_pavement"] > 0:
            zone_name = df[df["zone_id"] == zone_id]["name"].values[0]
            orig_temp = df[df["zone_id"] == zone_id]["lst_day_actual"].values[0]
            new_temp = df_current[df_current["zone_id"] == zone_id]["lst_day_actual"].values[0]
            hvi_val = df[df["zone_id"] == zone_id]["hvi"].values[0]
            cost = alloc["tree_planting"] * 80000.0 + alloc["green_roofs"] * 150000.0 + alloc["cool_pavement"] * 100000.0
            
            rec_list.append({
                "zone_id": zone_id, "name": zone_name, "hvi": float(hvi_val),
                "tree_planting": float(alloc["tree_planting"]), "green_roofs": float(alloc["green_roofs"]), "cool_pavement": float(alloc["cool_pavement"]),
                "original_temp": float(orig_temp), "optimized_temp": float(new_temp),
                "cooling_impact": float(orig_temp - new_temp), "cost": float(cost)
            })
            
    return allocation, sorted(rec_list, key=lambda x: x["cooling_impact"], reverse=True), total_spent

def get_2050_projection(df, model, rcp_scenario="rcp85"):
    df_proj = df.copy()
    
    if rcp_scenario == "rcp45":
        air_temp_increase = 1.5
        wind_speed_scale = 0.95
        solar_rad_scale = 0.98
    else:
        air_temp_increase = 3.2
        wind_speed_scale = 0.90
        solar_rad_scale = 0.96
        
    df_proj["air_temperature"] = df_proj["air_temperature"] + air_temp_increase
    df_proj["wind_speed"] = df_proj["wind_speed"] * wind_speed_scale
    df_proj["solar_radiation"] = df_proj["solar_radiation"] * solar_rad_scale
    
    for idx, row in df_proj.iterrows():
        if row["is_water"]: continue
        if row["ndvi"] > 0.25 and row["isf"] < 0.50:
            df_proj.loc[idx, "ndvi"] = np.clip(row["ndvi"] - 0.15, 0.05, 0.8)
            df_proj.loc[idx, "isf"] = np.clip(row["isf"] + 0.15, 0.1, 0.95)
            df_proj.loc[idx, "albedo"] = np.clip(row["albedo"] - 0.02, 0.05, 0.6)
        else:
            df_proj.loc[idx, "ndvi"] = np.clip(row["ndvi"] - 0.03, 0.05, 0.8)
            df_proj.loc[idx, "isf"] = np.clip(row["isf"] + 0.03, 0.1, 0.95)
        df_proj.loc[idx, "pop_density"] = np.clip(row["pop_density"] * 1.25, 500, 55000)
        
    # Recalculate physics & predict LST for land cells
    for idx, row in df_proj.iterrows():
        new_lst_phys = calculate_physics_lst(row["air_temperature"], row["solar_radiation"], row["albedo"], row["wind_speed"], row["ndvi"])
        df_proj.loc[idx, "lst_physics"] = new_lst_phys
        
    features = ["ndvi", "albedo", "isf", "pop_density", "wind_speed", "solar_radiation", "air_temperature"]
    land_mask = df_proj["is_water"] == False
    if land_mask.any():
        X_proj = df_proj.loc[land_mask, features]
        df_proj.loc[land_mask, "residual_pred"] = model.predict(X_proj)
        df_proj.loc[land_mask, "lst_day_pred"] = df_proj.loc[land_mask, "lst_physics"] + df_proj.loc[land_mask, "residual_pred"]
    
    df_proj.loc[~land_mask, "lst_day_pred"] = df_proj.loc[~land_mask, "lst_physics"]
    
    # Project Nighttime LST
    for idx, row in df_proj.iterrows():
        if row["is_water"]: continue
        cfg = CITY_CONFIGS[row["city"]]
        regional_air_temp_night = (cfg["base_temp"] - 12.0) + (air_temp_increase * 0.9)
        water_cool_factor = np.exp(-0.4 * row["distance_to_river"])
        df_proj.loc[idx, "lst_night_pred"] = (
            regional_air_temp_night + 
            4.5 * row["isf"] - 
            2.8 * row["ndvi"] - 
            1.1 * row["wind_speed"] + 
            1.8 * (row["pop_density"] / 30000.0) - 
            1.5 * water_cool_factor
        )
        
    return df_proj

if __name__ == "__main__":
    print("Initialising Multi-City UHI ML Pipeline...")
    dfs = []
    for city in CITY_CONFIGS.keys():
        print(f"Generating synthetic grid for {city.upper()}...")
        df = generate_city_dataset(city)
        dfs.append(df)
        
    combined_df = pd.concat(dfs, ignore_index=True)
    model, explainer, _ = train_physics_informed_model(combined_df)
    
    # Evaluate and calculate metrics per city
    for city, df in zip(CITY_CONFIGS.keys(), dfs):
        features = ["ndvi", "albedo", "isf", "pop_density", "wind_speed", "solar_radiation", "air_temperature"]
        df["residual_pred"] = 0.0
        
        # Apply XGBoost predictions to land cells
        land_mask = df["is_water"] == False
        if land_mask.any():
            df.loc[land_mask, "residual_pred"] = model.predict(df.loc[land_mask, features])
            
        df["lst_day_pred"] = df["lst_physics"] + df["residual_pred"]
        df = calculate_hvi_and_clusters(df)
        
        df.to_csv(f"{city}_urban_data_processed.csv", index=False)
        print(f"Saved processed grid file: {city}_urban_data_processed.csv")
    print("Multi-City analytical dataset generation completed successfully.")
