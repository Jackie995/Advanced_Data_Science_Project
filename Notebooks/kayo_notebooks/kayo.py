import pandas as pd
import zipfile

zip_path = "Data/kayo/ice_10yr_datasets.zip"
print("RUNNING MONTHLY VERSION")
# -------------------------
# REMOVALS
# -------------------------
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open("removals_10yr.csv") as f:
        df_removals = pd.read_csv(f)

df_removals["Departure Date"] = pd.to_datetime(
    df_removals["Departure Date"], errors="coerce"
)

df_removals = df_removals.dropna(subset=["Departure Date"])

df_removals["month"] = df_removals["Departure Date"].dt.to_period("M").astype(str)

removals_monthly = (
    df_removals.groupby("month")
    .size()
    .reset_index(name="removals")
)

# -------------------------
# ARRESTS
# -------------------------
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open("arrests_10yr.csv") as f:
        df_arrests = pd.read_csv(f)

df_arrests["Apprehension Date"] = pd.to_datetime(
    df_arrests["Apprehension Date"], errors="coerce"
)

df_arrests = df_arrests.dropna(subset=["Apprehension Date"])

df_arrests["month"] = df_arrests["Apprehension Date"].dt.to_period("M").astype(str)

arrests_monthly = (
    df_arrests.groupby("month")
    .size()
    .reset_index(name="arrests")
)

# -------------------------
# DETENTIONS
# -------------------------
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open("detentions_10yr.csv") as f:
        df_detentions = pd.read_csv(f)

df_detentions["Stay Book In Date"] = pd.to_datetime(
    df_detentions["Stay Book In Date"], errors="coerce"
)

df_detentions = df_detentions.dropna(subset=["Stay Book In Date"])

df_detentions["month"] = df_detentions["Stay Book In Date"].dt.to_period("M").astype(str)

detentions_monthly = (
    df_detentions.groupby("month")
    .size()
    .reset_index(name="detentions")
)

# -------------------------
# COMBINE ALL 3
# -------------------------
combined = (
    removals_monthly
    .merge(arrests_monthly, on="month", how="outer")
    .merge(detentions_monthly, on="month", how="outer")
)

# fill missing values with 0
combined = combined.fillna(0)

# make counts integers again
combined["removals"] = combined["removals"].astype(int)
combined["arrests"] = combined["arrests"].astype(int)
combined["detentions"] = combined["detentions"].astype(int)

# sort by month
combined = combined.sort_values("month").reset_index(drop=True)

print(combined.head())
print(combined.tail())
print(combined.shape)

combined.to_csv("Data/kayo/enforcement_monthly.csv", index=False)
print("saved enforcement_monthly.csv")
