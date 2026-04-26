import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ================================
# GRAPH 1: MODEL COMPARISON
# ================================
models = ["RF", "LR", "XGB", "LGBM", "CatBoost"]
r2_scores = [0.885, 0.55, 0.742, 0.752, 0.721]

plt.figure()
plt.bar(models, r2_scores)
plt.title("Model Comparison (R² Score)")
plt.xlabel("Models")
plt.ylabel("R²")
plt.savefig("model_comparison.png")   # saves image
plt.show()


# ================================
# GRAPH 2: LOSO PERFORMANCE
# ================================
stations = ["Hebbal", "Jigani", "Kasturi", "RVCE", "Shivapura", "Silk Board"]
r2_loso = [0.976, 0.644, 0.757, 0.883, 0.894, 0.965]

plt.figure()
plt.bar(stations, r2_loso)
plt.title("LOSO Performance (CatBoost)")
plt.xlabel("Stations")
plt.ylabel("R²")
plt.xticks(rotation=30)
plt.savefig("loso_performance.png")
plt.show()


# ================================
# GRAPH 3: LOSO LINE PLOT
# ================================
stations_line = ["Hebbal", "Jigani", "Kasturi", "RVCE", "Shivapura", "Silk Board"]
r2_line = [0.976, 0.644, 0.757, 0.883, 0.894, 0.965]

plt.figure()
plt.plot(stations_line, r2_line, marker='o')
plt.title("LOSO Performance Trend (CatBoost)")
plt.xlabel("Stations")
plt.ylabel("R²")
plt.xticks(rotation=30)
plt.savefig("loso_line_plot.png")
plt.show()

# ================================
# GRAPH 4: HEATMAP OVER GRID (from CSV)
# ================================
# Expect CSV with columns: latitude, longitude, value (e.g., PM2.5 or predictions)
try:
    df = pd.read_csv("hyperlocal_aqi_dataset_idw.csv")

    # If you have a column like PM2.5 or predictions, choose it
    value_col = "PM2.5" if "PM2.5" in df.columns else df.columns[-1]

    plt.figure()
    sc = plt.scatter(df["longitude"], df["latitude"], c=df[value_col], cmap="coolwarm", s=10)
    plt.colorbar(sc, label=value_col)
    plt.title("Spatial Heatmap (Grid Level)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("heatmap_grid.png")
    plt.show()

except Exception as e:
    print("Heatmap not generated. Ensure dataset has latitude, longitude columns.")
    print(e)