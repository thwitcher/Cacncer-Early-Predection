import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


class MedicalDataAnalysis():
    def __init__(self):
        self.data = None

        
    def import_data(self):
        self.data = pd.read_csv(r'C:\Users\borch\Desktop\projet_AFD\cancer_des_poumons.csv')
        print(self.data)

    def jeu_donnees_data(self):
        if self.data is not None:
            nombre_observations, nombre_caracteristiques = self.data.shape
            message = "Nombre d'observations : {}\nNombre de caractéristiques : {}".format(nombre_observations, nombre_caracteristiques)
            print(message)
        else:
            print("No data imported.")

        
    def replace_Nan_data(self):
        if self.data is not None:
            nan_count = self.data.isnull().sum().sum()
            if nan_count > 0:
                message = "There are {} missing values. Missing values replaced with column means.".format(nan_count)
                print(message)
            else:
                print("No missing values in the data.")
        else:
            print("No data imported.")
        
    def codage_data(self):
        if self.data is not None:
            self.data['GENDER'] = self.data['GENDER'].map({'F': 0, 'M': 1})
            print("Categorical variables encoded successfully.")
        else:
            print("No data imported.")

    def check_normalazied_data(self):
        scaler = StandardScaler()
        self.data= pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        print("Data normalized")

    def preprocess_data(self):
        if self.data is not None:
            self.data = self.data.fillna(self.data.mean())
            scaler = StandardScaler()
            self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
            print("Data preprocessing completed.")
        else:
            print("No data imported.")
    def correlation_matrix(self):
        correlation_matrix = self.data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()

        # Find most correlated pairs of variables
        max_corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)
        max_corr_pairs = max_corr_pairs[max_corr_pairs != 1].head(5)  
        message = "Most correlated variable pairs:\n{}".format(max_corr_pairs)
        print(message)

    def apply_pca(self):
        if self.data is not None:
            pca = PCA()
            pca.fit(self.data)
           
            eigenvalues = pca.explained_variance_
                    
# Pourcentage d'inertie expliqué par chaque composante principale
            explained_variance_ratio = pca.explained_variance_ratio_

            eigenvalues_message = "Valeurs propres:\n{}".format(eigenvalues)
            explained_variance_message = "Pourcentage d'inertie expliqué par chaque composante principale:\n{}".format(explained_variance_ratio)

            print(eigenvalues_message)
            print(explained_variance_message)

# Affichage de l'éboulis des valeurs propres
            plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
            plt.xlabel('Composantes principales')
            plt.ylabel('Valeurs propres')
            plt.title('Éboulis des valeurs propres')
            plt.grid(True)
            plt.show()
        else:
            print("No data imported.")
        
        loadings = pca.components_.T * np.sqrt(eigenvalues)

# Affichage de la saturation des variables
        plt.figure(figsize=(10, 8))
        for i, (x, y) in enumerate(zip(loadings[:, 0], loadings[:, 1])):
            plt.plot([0, x], [0, y], color='k')
            plt.text(x, y, self.data.columns[i], fontsize='12', ha='center', va='center')
            plt.scatter(x, y, marker='o', color='b', s=100)  # Plot points on the circle
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Cercle de corrélation')
        plt.grid(True)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

        circle = plt.Circle((0,0), radius=1, color='b', fill=False)
        plt.gca().add_patch(circle)
        plt.show()
    def apply_clustering(self):
        if self.data is not None:
            kmeans = KMeans(n_clusters=2, random_state=42)
            hac = AgglomerativeClustering(n_clusters=2)
            scaler = StandardScaler()
            Z = scaler.fit_transform(self.data)
            pca = PCA()     
            Y = pca.fit_transform(Z)
            kmeans.fit(Y)

        # Fit both clustering algorithms to the data
            kmeans.fit(self.data)
            hac.fit(self.data)

        # Calculate silhouette scores for both algorithms
            silhouette_kmeans = silhouette_score(self.data, kmeans.labels_)
            silhouette_hac = silhouette_score(self.data, hac.labels_)

        # Print silhouette scores
            silhouette_kmeans_message = "Silhouette Score for K-means: {}".format(silhouette_kmeans)
            silhouette_hac_message = "Silhouette Score for Hierarchical Agglomerative Clustering: {}".format(silhouette_hac)

            print(silhouette_kmeans_message)
            print(silhouette_hac_message)

        # Determine the best clustering algorithm
            if silhouette_kmeans > silhouette_hac:
                best_cluster = kmeans.labels_
                best_algorithm = "K-means"
            else:
                best_cluster = hac.labels_
                best_algorithm = "Hierarchical Agglomerative Clustering"

        # Add the best cluster labels to the DataFrame
            self.data['Best Cluster'] = best_cluster

        # Display the counts of samples in the best cluster
            best_cluster_message = "\nBest Cluster Assignments ({}):\n{}".format(best_algorithm, self.data['Best Cluster'].value_counts())
            print(best_cluster_message)

        else:
            print("No data imported.")
    def compare_clustering(self):
        # Standardiser les données
        scaler = StandardScaler()
        Z = scaler.fit_transform(self.data)
        
        # Réduire les dimensions avec PCA
        pca = PCA()
        Y = pca.fit_transform(Z)
        
        # K-means
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(Y)
        kmeans_labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        # Afficher les coordonnées de chaque centroïde et l’inertie associée
        print("Coordonnées des centroïdes :")
        for i, centroid in enumerate(centroids):
            print(f"Cluster {i+1}: {centroid}")
            print("Inertie associée :", kmeans.inertia_)
        


        # Afficher les étiquettes des individus en sortie ainsi que le nombre d’individus de chaque classe
        print("Étiquettes des individus :", kmeans_labels)
        print("Nombre d'individus de chaque classe :", np.bincount(kmeans_labels))

# Représenter graphiquement Y ainsi que les centres des clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=kmeans_labels, palette="Set1", s=100)
        sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], color='black', marker='x', s=200, label='Centroids')
        plt.title("ACP avec clustering KMeans (2 clusters)")
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.legend()
        plt.show()

# Afficher les distances des individus aux centres des clusters
        distances = kmeans.transform(Y)
        print("Distances des individus aux centres des clusters :", distances)

# Evaluer la qualité des regroupements lorsque k varie entre 2 et 6 en se basant sur la méthode de coude (Elbow Method)
# puis sur la méthode de silhouette. Déduire K optimal
        inertia_values = []
        silhouette_scores = []

        for k in range(2, 7):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(Y)
            inertia_values.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(Y, kmeans.labels_))

# Méthode de coude (Elbow Method)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(2, 7), inertia_values, marker='o')
        plt.title("Méthode de coude")
        plt.xlabel("Nombre de clusters")
        plt.ylabel("Inertie")

# Méthode de silhouette
        plt.subplot(1, 2, 2)
        plt.plot(range(2, 7), silhouette_scores, marker='o')
        plt.title("Méthode de silhouette")
        plt.xlabel("Nombre de clusters")
        plt.ylabel("Score de silhouette")
        plt.show()
        
        
        
        
        
        
        
        
        # CAH
        cah = AgglomerativeClustering(n_clusters=2)
        cah.fit(self.data)
        if hasattr(cah, 'labels_'):
            cah_labels = cah.labels_
        else:
            cah_labels = cah.fit_predict(self.data)

        
        # Visualisation des clusters pour K-means et CAH
        plt.figure(figsize=(12, 6))
        
        # Visualisation des clusters pour K-means
        plt.subplot(1, 2, 1)
        plt.scatter(Y[:, 0], Y[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='red', label='Centroids')
        plt.title('K-means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()

        # Visualisation des clusters pour CAH
        plt.subplot(1, 2, 2)
        plt.scatter(Y[:, 0], Y[:, 1], c=cah_labels, cmap='viridis', alpha=0.5)
        plt.title('Agglomerative Clustering (CAH)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        plt.show()

    
    # Apply hierarchical clustering algorithm
        Z = linkage(self.data, method='ward')

    # Plot dendrogram
        plt.figure(figsize=(10, 5))
        plt.title('Dendrogram')
        dendrogram(Z)
        plt.show()


if __name__ == "__main__":
    # Instantiate the MedicalDataAnalysis class
    app = MedicalDataAnalysis()
    
    # Call the methods as needed
    app.import_data()
    app.jeu_donnees_data()
    app.replace_Nan_data()
    app.codage_data()
    app.check_normalazied_data()
    app.preprocess_data()
    app.correlation_matrix()
    app.apply_pca()
    app.apply_clustering()
    app.compare_clustering()
