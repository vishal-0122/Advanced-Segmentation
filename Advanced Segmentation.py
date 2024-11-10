#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[54]:


df = pd.read_csv("Mall_Customers.csv")


# In[55]:


df.head()


# In[56]:


data = df.drop('CustomerID', axis=1)

# Encode the Gender column
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

data.head()


# In[57]:


scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)

# Convert back to a DataFrame for easy interpretation
processed_data = pd.DataFrame(scaled_features, columns=data.columns)

processed_data.head()


# In[51]:


# Select features for clustering
X_Gender = processed_data[['Gender', 'Spending Score (1-100)']]


# In[58]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
processed_data['Cluster'] = kmeans.fit_predict(X_Gender)


# In[59]:


plt.figure(figsize=(17, 10))
plt.scatter(X_Gender['Gender'], X_Gender['Spending Score (1-100)'], c=processed_data['Cluster'], cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.80, marker='X')
plt.title("Customer Segmentation using KMeans")
plt.xlabel("Gender(scaled)")
plt.ylabel("Spending Score (scaled)")
plt.grid(True)
plt.show()


# In[61]:


# Applying elbow method for Age feature
X_Age = processed_data[['Age', 'Spending Score (1-100)']]
inertia = []

# Test a range of k values (e.g., from 1 to 10 clusters)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_Age)  # Fit the model on X_Age
    inertia.append(kmeans.inertia_)  # Append inertia for each k

# Plot the Elbow curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()


# In[62]:


optim_k = 2
X_Age = processed_data[['Age', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters = optim_k, random_state=42)
processed_data['Cluster'] = kmeans.fit_predict(X_Age)


# In[63]:


plt.figure(figsize=(17, 10))
plt.scatter(X_Age['Age'], X_Age['Spending Score (1-100)'], c=processed_data['Cluster'], cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.80, marker='X')
plt.title("Customer Segmentation using KMeans")
plt.xlabel("Age (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.grid(True)
plt.show()


# In[64]:


# Applying elbow method for Annual Income feature
X_AnnualIn = processed_data[['Annual Income (k$)', 'Spending Score (1-100)']]
inertia = []

# Test a range of k values (e.g., from 1 to 10 clusters)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_AnnualIn)  # Fit the model on X_Age
    inertia.append(kmeans.inertia_)  # Append inertia for each k

# Plot the Elbow curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()


# In[65]:


optim_k = 5
X_Age = processed_data[['Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters = optim_k, random_state=42)
processed_data['Cluster'] = kmeans.fit_predict(X_Age)


# In[66]:


plt.figure(figsize=(17, 10))
plt.scatter(X_AnnualIn['Annual Income (k$)'], X_AnnualIn['Spending Score (1-100)'], c=processed_data['Cluster'], cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.80, marker='X')
plt.title("Customer Segmentation using KMeans")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.grid(True)
plt.show()


# In[67]:


plt.figure(figsize=(15, 10))
for i, col in enumerate(data.columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='Cluster', y=col, data=processed_data, palette='viridis')
    plt.title(f"Box Plot of {col} by Cluster")
plt.tight_layout()
plt.show()


# In[ ]:




