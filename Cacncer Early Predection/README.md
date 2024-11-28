# Medical Data Analysis Project

**A Python project for preprocessing, analyzing, and clustering medical data.**

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Technologies Used](#technologies-used)
6. [Output Visualizations](#output-visualizations)
7. [License](#license)

---

## 📖 Overview
This project is focused on analyzing medical data (e.g., lung cancer dataset). It includes steps for data preprocessing, applying Principal Component Analysis (PCA), and clustering using K-means and Hierarchical Agglomerative Clustering (HAC).

---

## ✨ Features
- Import and preprocess medical datasets
- Replace missing values and encode categorical data
- Normalize and scale datasets
- Visualize correlations using heatmaps
- Perform PCA for dimensionality reduction
- Cluster data with K-means and HAC
- Compare clustering algorithms using silhouette scores
- Generate dendrograms for hierarchical clustering

---

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/medical-data-analysis.git
   cd medical-data-analysis

Setup
Prerequisites
Ensure the following Python libraries are installed:

pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
Install the dependencies via pip:

Install dependencies: Ensure you have Python installed. Then, run:

bash
Copier le code
pip install -r requirements.txt
Sample requirements.txt:

text
Copier le code
pandas
matplotlib
seaborn
scikit-learn
scipy
numpy
Add your dataset: Place the dataset file (cancer_des_poumons.csv) in the project folder.

🛠️ Usage
Run the script:
 Install dependencies: Ensure you have Python installed. Then, run:

bash
Copier le code
pip install -r requirements.txt

Sample requirements:

text
Copier le code
pandas
matplotlib
seaborn
scikit-learn
scipy
numpy
Add your dataset: Place the dataset file (cancer_des_poumons.csv) in the project folder.

## 🛠️ Usage
Run the script:

bash
Copier le code
python main.py
Available Methods:

import_data(): Import data from the dataset
jeu_donnees_data(): Display dataset dimensions
replace_Nan_data(): Replace missing values
correlation_matrix(): Generate and visualize a correlation matrix
apply_pca(): Perform PCA and plot eigenvalues
apply_clustering(): Apply K-means and HAC
compare_clustering(): Compare clustering methods
Customize the script as needed: The main.py script is preconfigured to run all methods sequentially.

## 📊 Output Visualizations
` Presentation.pdf `

 ## 🖥️ Technologies Used
Python for data analysis
Pandas for data manipulation
Matplotlib and Seaborn for visualizations
Scikit-learn for PCA and clustering
SciPy for hierarchical clustering
 ## 📜 License
Use this project freely 

## 👩‍💻 Author
Developed by Your Abderrahmen Borchani.

For inquiries, reach out at borchani.abderrahmen25@gmail.com.
