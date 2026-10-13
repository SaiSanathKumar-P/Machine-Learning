# Machine Learning Experiments (Revised & Validated)

This repository contains the complete, technically reviewed, and validated suite of 16 Machine Learning laboratory experiments.

## Project Structure

```
Machine_Learning-Revised/
├── Decision Tree Classifier/
│   ├── Q1_Classification_Tree.ipynb
│   └── Q2_Weather_Classification_Tree.ipynb
├── K Nearest Neighbors/
│   ├── Ex1_Handwritten_Digit_kNN.ipynb
│   └── Ex2_Euclidean_Distance.ipynb
├── K_Means_Clustering/
│   ├── 1_KMeans_Old_Faithful_Geyser.ipynb
│   ├── 2_KMeans_Image_Compression.ipynb
│   └── compressed_image_kmeans.png
├── Linear Regression/
│   ├── Exercise_1_Chicago_Taxi.ipynb
│   └── Exercise_2_Body_Weight.ipynb
├── Logistic Regression/
│   ├── Exercise_1_Graduate_Admission.ipynb
│   ├── Exercise_2_Disease.ipynb
│   └── Exercise_3_Online_Purchase.ipynb
├── Principal Component Analysis/
│   ├── Ex1_Real_Estate_Unsold_Analysis.ipynb
│   └── Ex2_PCA_Principal_Component.ipynb
├── Random_Forests/
│   └── 1_Random_forest_fruit_basket.ipynb
├── Support Vector Machines/
│   ├── Ex1_SVM_Quadratic_Distribution.ipynb
│   └── Ex2_SVM_SMS_Spam.ipynb
├── data/
│   ├── binary.dta
│   ├── chicago_taxi_train.csv
│   ├── Howell1.csv
│   ├── online_shoppers_intention.csv
│   ├── pima-indians-diabetes.data.csv
│   └── SMSSpamCollection
├── requirements.txt
└── README.md
```

## Key Upgrades in the Revised Edition

1. **Elimination of Data Leakage:**
   - Preprocessing transformers (`TfidfVectorizer`, `OneHotEncoder`, `StandardScaler`) are encapsulated in scikit-learn `Pipeline` and `ColumnTransformer` constructs, fitted strictly on training data splits.
2. **Missing Value & Artifact Treatment:**
   - In Pima Indians Diabetes, physiologically impossible zeroes (Glucose, Blood Pressure, Insulin, BMI) are audited, converted to `NaN`, and imputed using training-set medians.
3. **Full Algorithm Implementation:**
   - In PCA Experiment 1 (Real Estate), the previously missing PCA was implemented from scratch (standardization, SVD, scree plot, factor loadings, and 2D biplot).
4. **Decoupling from Cloud/Colab APIs:**
   - Replaced Google Colab interactive upload calls with cross-platform local asset handling and self-contained execution.
5. **100% Offline Reproducibility:**
   - All external datasets are cached in `data/`, with dual local/remote fallbacks.

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter
```bash
jupyter lab
# or
jupyter notebook
```
Select **Kernel -> Restart & Run All** on any notebook. All 16 notebooks are tested and run without errors.
