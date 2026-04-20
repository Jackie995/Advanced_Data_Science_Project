import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# -------------------------
# PATHS
# -------------------------
data_path = "Data/kayo/enforcement_monthly.csv"
output_plot_dir = "Notebooks/kayo_notebooks/plots"
output_clustered_csv = "Data/kayo/enforcement_monthly_clustered.csv"

# make sure plot folder exists
os.makedirs(output_plot_dir, exist_ok=True)


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(data_path)

print("First 5 rows:")
print(df.head())
print("\nShape:")
print(df.shape)
print("\nColumns:")
print(df.columns)


# -------------------------
# PREP FEATURES
# -------------------------
features = df[["removals", "arrests", "detentions"]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# -------------------------
# ELBOW METHOD
# -------------------------
inertia = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.tight_layout()
plt.savefig(f"{output_plot_dir}/elbow_method.png")
plt.close()

print("\nSaved elbow plot to Notebooks/kayo_notebooks/plots/elbow_method.png")


# -------------------------
# SILHOUETTE SCORES
# -------------------------
print("\nSilhouette Scores:")
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    print(f"k={k}, silhouette score={score:.4f}")


# -------------------------
# FINAL KMEANS MODEL
# -------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(scaled_features)


# -------------------------
# CLUSTER SUMMARY
# -------------------------
print("\nCluster Averages:")
cluster_summary = df.groupby("cluster")[["removals", "arrests", "detentions"]].mean()
print(cluster_summary)

print("\nCluster Counts:")
print(df["cluster"].value_counts().sort_index())


# -------------------------
# CLUSTER OVER TIME
# -------------------------
df["month_dt"] = pd.to_datetime(df["month"])

plt.figure(figsize=(12, 5))
plt.scatter(df["month_dt"], df["cluster"])
plt.xlabel("Month")
plt.ylabel("Cluster")
plt.title("Cluster Assignment Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_plot_dir}/cluster_over_time.png")
plt.close()

print("Saved time plot to Notebooks/kayo_notebooks/plots/cluster_over_time.png")


# -------------------------
# CLUSTER SCATTER PLOT
# -------------------------
plt.figure(figsize=(8, 5))
plt.scatter(df["removals"], df["arrests"], c=df["cluster"])
plt.xlabel("Removals")
plt.ylabel("Arrests")
plt.title("Clusters of Enforcement Behavior")
plt.tight_layout()
plt.savefig(f"{output_plot_dir}/cluster_scatter.png")
plt.close()

print("Saved scatter plot to Notebooks/kayo_notebooks/plots/cluster_scatter.png")


# -------------------------
# OPTIONAL: REMOVALS VS DETENTIONS
# -------------------------
plt.figure(figsize=(8, 5))
plt.scatter(df["removals"], df["detentions"], c=df["cluster"])
plt.xlabel("Removals")
plt.ylabel("Detentions")
plt.title("Removals vs Detentions by Cluster")
plt.tight_layout()
plt.savefig(f"{output_plot_dir}/removals_vs_detentions.png")
plt.close()

print("Saved removals vs detentions plot to Notebooks/kayo_notebooks/plots/removals_vs_detentions.png")


# -------------------------
# SAVE CLUSTERED DATA
# -------------------------
df.to_csv(output_clustered_csv, index=False)
print(f"\nSaved clustered dataset to {output_clustered_csv}")
