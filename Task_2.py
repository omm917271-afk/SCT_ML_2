# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset from GitHub
df = pd.read_csv(
    "https://raw.githubusercontent.com/sharmaroshan/Clustering-of-Mall-Customers/master/Mall_Customers.csv")

# Basic info
print(df.head())

# Rename 'Genre' to 'Gender' for consistency
df.rename(columns={'Genre': 'Gender'}, inplace=True)

# Drop CustomerID as it is not useful for clustering
df.drop('CustomerID', axis=1, inplace=True)

# Encode Gender
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

# Select features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.grid()
plt.show()

# Apply KMeans with chosen K (let’s assume k=5 from Elbow)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Silhouette Score (optional for validation)
score = silhouette_score(X_scaled, df['cluster'])
print("Silhouette Score:", score)

# Visualization (Annual income vs Spending Score)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'],
                hue=df['cluster'], palette='Set2', s=100)
plt.title("Customer Segments (by Income vs Spending Score)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.grid()
plt.show()

# Cluster summary
cluster_summary = df.groupby('cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("Cluster Summary:\n", cluster_summary)