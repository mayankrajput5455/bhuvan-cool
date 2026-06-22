import pandas as pd
import pickle
import model_pipeline as mp
import traceback

df = pd.read_csv("lucknow_urban_data_processed.csv")
with open("xgb_residual_model.pkl", "rb") as f:
    model = pickle.load(f)

active_interventions = {
    row["zone_id"]: {"tree_planting": 0, "green_roofs": 0, "cool_pavement": 0}
    for _, row in df.iterrows()
}
active_interventions["zone_19"] = {"tree_planting": 20, "green_roofs": 25, "cool_pavement": 25}

try:
    df_sim, cost = mp.run_simulation_inference(df, model, active_interventions)
    print("Success!")
except Exception as e:
    traceback.print_exc()
