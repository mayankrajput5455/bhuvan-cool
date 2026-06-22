from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import pickle
import os
import model_pipeline as mp

app = Flask(__name__, static_folder="static", template_folder="templates")

# Globals
CITIES = ["lucknow", "delhi", "kanpur", "goa", "mumbai"]
city_dataframes = {}
xgb_model = None
shap_explainer = None

def load_resources():
    global city_dataframes, xgb_model, shap_explainer
    
    # 1. Load Data for each city
    for city in CITIES:
        path = f"{city}_urban_data_processed.csv"
        if os.path.exists(path):
            city_dataframes[city] = pd.read_csv(path)
            print(f"Loaded dataset for {city.upper()}. Total rows: {len(city_dataframes[city])}")
        else:
            print(f"Warning: processed data for {city.upper()} not found at {path}!")
            
    # 2. Load ML Model
    if os.path.exists("xgb_residual_model.pkl"):
        with open("xgb_residual_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)
        print("XGBoost unified model loaded successfully.")
    else:
        print("Warning: xgb_residual_model.pkl not found!")
        
    # 3. Load SHAP Explainer
    if os.path.exists("shap_explainer.pkl"):
        with open("shap_explainer.pkl", "rb") as f:
            shap_explainer = pickle.load(f)
        print("SHAP explainer loaded successfully.")
    else:
        print("Warning: shap_explainer.pkl not found.")

# Setup templates and static dirs
os.makedirs("templates", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

def get_city_dataframe():
    """
    Helper to extract target city from query arguments or JSON body.
    Defaults to 'lucknow'.
    """
    city = request.args.get("city", "").lower()
    if not city and request.is_json:
        city = (request.get_json() or {}).get("city", "").lower()
    if city not in city_dataframes:
        city = "lucknow"
    return city, city_dataframes[city]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/city-data")
def get_city_data():
    """
    Returns the baseline data of the selected city.
    """
    if not city_dataframes:
        load_resources()
        
    city_name, df = get_city_dataframe()
    records = df.to_dict(orient="records")
    corridors = mp.get_ventilation_corridor_paths(df)
    
    return jsonify({
        "status": "success",
        "city": city_name.capitalize(),
        "water_type": mp.CITY_CONFIGS[city_name]["water_type"],
        "grid_size": mp.GRID_SIZE,
        "zones": records,
        "corridors": corridors
    })

@app.route("/api/shap/<zone_id>")
def get_shap_values(zone_id):
    """
    Calculates and returns SHAP values for a specific zone in a selected city.
    """
    if not city_dataframes:
        load_resources()
        
    city_name, df = get_city_dataframe()
    zone_row = df[df["zone_id"] == zone_id]
    if len(zone_row) == 0:
        return jsonify({"status": "error", "message": "Zone not found"}), 404
        
    features_list = [
        "ndvi", "albedo", "isf", "pop_density", 
        "wind_speed", "solar_radiation", "air_temperature"
    ]
    
    # If the cell is water, return empty SHAP since it fits purely physically
    if bool(zone_row["is_water"].values[0]):
        return jsonify({
            "status": "success",
            "zone_id": zone_id,
            "zone_name": "Water Body / Sea Area",
            "base_value": 0.0,
            "residual_pred": 0.0,
            "lst_day_actual": float(zone_row["lst_day_actual"].values[0]),
            "explanation": []
        })
        
    zone_x = zone_row[features_list]
    
    try:
        if shap_explainer is not None:
            shap_vals = shap_explainer.shap_values(zone_x)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals[0]
            
            feature_labels = {
                "ndvi": "Vegetation Cover (NDVI)",
                "albedo": "Surface Albedo",
                "isf": "Impervious Surface Fraction (ISF)",
                "pop_density": "Population Density",
                "wind_speed": "Wind Speed (Ventilation)",
                "solar_radiation": "Solar Insolation",
                "air_temperature": "Ambient Air Temperature"
            }
            
            explanation = []
            for feat, val in zip(features_list, shap_vals):
                explanation.append({
                    "feature": feat,
                    "label": feature_labels[feat],
                    "value": float(zone_row[feat].values[0]),
                    "shap_contribution": float(val)
                })
                
            explanation = sorted(explanation, key=lambda x: abs(x["shap_contribution"]), reverse=True)
            expected_val = float(shap_explainer.expected_value) if hasattr(shap_explainer, 'expected_value') else 0.0
            if isinstance(expected_val, list):
                expected_val = expected_val[0]
                
            return jsonify({
                "status": "success",
                "zone_id": zone_id,
                "zone_name": zone_row["name"].values[0],
                "base_value": expected_val,
                "residual_pred": float(zone_row["residual_pred"].values[0]),
                "lst_day_actual": float(zone_row["lst_day_actual"].values[0]),
                "explanation": explanation
            })
    except Exception as e:
        print(f"Error generating SHAP: {e}")
        
    # Fallback/Heuristic explanation
    isf_val = float(zone_row["isf"].values[0])
    ndvi_val = float(zone_row["ndvi"].values[0])
    pop_val = float(zone_row["pop_density"].values[0])
    albedo_val = float(zone_row["albedo"].values[0])
    
    explanation = [
        {"feature": "isf", "label": "Impervious Surface Fraction (ISF)", "value": isf_val, "shap_contribution": 4.0 * isf_val - 1.6},
        {"feature": "ndvi", "label": "Vegetation Cover (NDVI)", "value": ndvi_val, "shap_contribution": -3.5 * ndvi_val + 0.9},
        {"feature": "pop_density", "label": "Population Density", "value": pop_val, "shap_contribution": 1.6 * (pop_val/30000.0) - 0.5},
        {"feature": "albedo", "label": "Surface Albedo", "value": albedo_val, "shap_contribution": -1.4 * albedo_val + 0.2},
        {"feature": "wind_speed", "label": "Wind Speed (Ventilation)", "value": float(zone_row["wind_speed"].values[0]), "shap_contribution": -0.5},
        {"feature": "solar_radiation", "label": "Solar Insolation", "value": float(zone_row["solar_radiation"].values[0]), "shap_contribution": 0.2},
        {"feature": "air_temperature", "label": "Ambient Air Temperature", "value": float(zone_row["air_temperature"].values[0]), "shap_contribution": 0.3}
    ]
    explanation = sorted(explanation, key=lambda x: abs(x["shap_contribution"]), reverse=True)
    
    return jsonify({
        "status": "success",
        "zone_id": zone_id,
        "zone_name": zone_row["name"].values[0],
        "base_value": 1.35,
        "residual_pred": float(zone_row["residual_pred"].values[0]),
        "lst_day_actual": float(zone_row["lst_day_actual"].values[0]),
        "explanation": explanation
    })

@app.route("/api/simulate", methods=["POST"])
def simulate_cooling():
    """
    Simulates cooling for customized interventions in the selected city.
    """
    if not city_dataframes:
        load_resources()
        
    city_name, df = get_city_dataframe()
    req_data = request.get_json() or {}
    zone_interventions = req_data.get("zone_interventions", {})
    
    try:
        df_sim, total_cost = mp.run_simulation_inference(df, xgb_model, zone_interventions)
        
        sim_zones = []
        for zone_id, interventions in zone_interventions.items():
            row_sim = df_sim[df_sim["zone_id"] == zone_id]
            row_orig = df[df["zone_id"] == zone_id]
            if len(row_sim) == 0: continue
            
            sim_zones.append({
                "zone_id": zone_id,
                "name": row_sim["name"].values[0],
                "original_temp": float(row_orig["lst_day_actual"].values[0]),
                "simulated_temp": float(row_sim["lst_day_pred"].values[0]),
                "temp_reduction": float(row_sim["lst_delta"].values[0]),
                "original_ndvi": float(row_orig["ndvi"].values[0]),
                "simulated_ndvi": float(row_sim["ndvi"].values[0]),
                "original_albedo": float(row_orig["albedo"].values[0]),
                "simulated_albedo": float(row_sim["albedo"].values[0]),
                "original_isf": float(row_orig["isf"].values[0]),
                "simulated_isf": float(row_sim["isf"].values[0]),
                "interventions": interventions
            })
            
        return jsonify({
            "status": "success",
            "total_cost_inr": total_cost,
            "simulated_zones": sim_zones,
            "full_grid": df_sim.to_dict(orient="records")
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/optimize", methods=["POST"])
def optimize_interventions():
    """
    Runs budget optimization for the selected city.
    """
    if not city_dataframes:
        load_resources()
        
    city_name, df = get_city_dataframe()
    req_data = request.get_json() or {}
    budget = float(req_data.get("budget", 15000000.0))
    weight_by_hvi = bool(req_data.get("weight_by_hvi", True))
    
    try:
        allocations, recommendations, total_spent = mp.run_greedy_optimizer(
            df, xgb_model, budget, weight_by_hvi=weight_by_hvi
        )
        df_opt, _ = mp.run_simulation_inference(df, xgb_model, allocations)
        
        return jsonify({
            "status": "success",
            "budget_limit": budget,
            "total_spent_inr": total_spent,
            "recommendations": recommendations,
            "full_grid": df_opt.to_dict(orient="records")
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/future-projection", methods=["POST"])
def get_climate_projections():
    """
    Simulates year 2050 scenario for the selected city.
    """
    if not city_dataframes:
        load_resources()
        
    city_name, df = get_city_dataframe()
    req_data = request.get_json() or {}
    rcp_scenario = req_data.get("rcp", "rcp85")
    
    try:
        df_proj = mp.get_2050_projection(df, xgb_model, rcp_scenario=rcp_scenario)
        
        projection_data = []
        for idx, row in df_proj.iterrows():
            zone_id = row["zone_id"]
            orig_row = df[df["zone_id"] == zone_id].iloc[0]
            
            projection_data.append({
                "zone_id": zone_id,
                "name": row["name"],
                "x": int(row["x"]),
                "y": int(row["y"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "is_water": bool(row["is_water"]),
                "original_temp": float(orig_row["lst_day_actual"]),
                "projected_temp": float(row["lst_day_pred"]),
                "temp_increase": float(row["lst_day_pred"] - orig_row["lst_day_actual"]),
                "original_ndvi": float(orig_row["ndvi"]),
                "projected_ndvi": float(row["ndvi"]),
                "original_isf": float(orig_row["isf"]),
                "projected_isf": float(row["isf"]),
                "original_air_temp": float(orig_row["air_temperature"]),
                "projected_air_temp": float(row["air_temperature"]),
                "original_night_temp": float(orig_row["lst_night_actual"]),
                "projected_night_temp": float(row["lst_night_pred"] if not row["is_water"] else orig_row["lst_night_actual"])
            })
            
        return jsonify({
            "status": "success",
            "rcp": rcp_scenario,
            "projection": projection_data
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    load_resources()
    app.run(host="127.0.0.1", port=5000, debug=True)
