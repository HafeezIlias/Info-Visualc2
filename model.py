# %% [markdown]
# # CO2 Emissions Analysis and Prediction Model
# 
# This notebook is organized into the following sections:
# 1. Imports & Settings  
# 2. Utility Functions  
# 3. Data Loading  
# 4. Preprocessing  
# 5. Exploratory Data Analysis  
# 6. Clustering Analysis  
# 7. Feature Selection  
# 8. Model Training & Evaluation with Hyperparameter Tuning (7 Models)
# 9. Main Execution

# %%
# ====================
# 1. IMPORTS & SETTINGS
# ====================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # Add pickle for model export

from scipy.stats import describe, uniform, randint
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    roc_auc_score
)

# limit threads for reproducibility
os.environ["OMP_NUM_THREADS"] = "1"

# %% [markdown]
# ## 2. Utility Functions  
# Helper functions for statistics, plotting, and evaluation.

# %%
def print_distribution_stats(data: pd.Series, name: str):
    """Print basic stats and an ASCII histogram for a numeric feature."""
    stats = describe(data.values)
    q = np.percentile(data, [25, 50, 75])
    print(f"\n--- {name} ---")
    print(f"Count: {stats.nobs}, Mean: {stats.mean:.2f}, Median: {q[1]:.2f}")
    print(f"Std: {np.sqrt(stats.variance):.2f}, Min: {stats.minmax[0]:.2f}, Max: {stats.minmax[1]:.2f}")
    print(f"25th/50th/75th: {q.round(2).tolist()}")
    print(f"Skewness: {stats.skewness:.2f}, Kurtosis: {stats.kurtosis:.2f}")
    hist, _ = np.histogram(data, bins=10)
    mx = hist.max()
    for count in hist:
        bar = '█' * int((count/mx)*20)
        print(f"{bar:20s} | {count}")
    print()


def evaluate_classifier(model, X_test, y_test, name: str):
    """Print metrics, confusion matrix, and ROC (if binary)."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n*** {name} ***")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC curve (binary only)
    if len(np.unique(y_test)) == 2:
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
        plt.plot([0,1],[0,1],'k--')
        plt.title(f"{name} ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.show()

# %% [markdown]
# ## 3. Data Loading  

# %%
def load_data(path: str) -> pd.DataFrame:
    """Load CSV and encode categorical fields."""
    df = pd.read_csv(path)
    df['Industry_Type_Encoded'] = LabelEncoder().fit_transform(df['Industry_Type'])
    df['Continent_Encoded']    = LabelEncoder().fit_transform(df['Continent'])
    return df

# %% [markdown]
# ## 4. Preprocessing  

# %%
def preprocess(df: pd.DataFrame, features: list):
    """Scale features and split into train/test."""
    X = df[features].values
    y = df['Continent_Encoded'].values
    X_scaled = StandardScaler().fit_transform(X)
    return train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

# %% [markdown]
# ## 5. Exploratory Data Analysis  

# %%
def exploratory_analysis(df: pd.DataFrame, features: list):
    """Print distribution stats and top correlations."""
    print("\n=== EDA: Distribution Statistics ===")
    for f in features:
        print_distribution_stats(df[f], f)
    corr = df[features].corr().abs()
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    print("Top 5 feature correlations:\n", pairs.head(5), "\n")

# %% [markdown]
# ## 6. Clustering Analysis  

# %%
def cluster_analysis(X_scaled: np.ndarray):
    """K-Means clustering metrics and PCA scatter."""
    print("\n=== K-Means Clustering Metrics ===")
    km = KMeans(n_clusters=2, random_state=42, n_init='auto')
    labels = km.fit_predict(X_scaled)
    print(f"Silhouette: {silhouette_score(X_scaled, labels):.3f}")
    print(f"Calinski-Harabasz: {calinski_harabasz_score(X_scaled, labels):.3f}")
    print(f"Davies-Bouldin: {davies_bouldin_score(X_scaled, labels):.3f}\n")
    pca_proj = PCA(n_components=2).fit_transform(X_scaled)
    plt.scatter(pca_proj[:,0], pca_proj[:,1], c=labels, cmap='viridis', alpha=0.6)
    plt.title("K-Means (PCA Projection)")
    plt.xlabel("PC1"), plt.ylabel("PC2")
    plt.show()

# %% [markdown]
# ## 7. Feature Selection  

# %%
def select_top_features(X_train, y_train, feature_names, top_n=5):
    """Train Decision Tree and return top_n features by importance."""
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    importances = dt.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_names[i] for i in idx]
    top_importances = importances[idx]
    
    # Visualize top features
    plt.figure(figsize=(12, 6))
    plt.bar(top_feats, top_importances)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title(f'Top {top_n} Most Important Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print(f"Top {top_n} features:", top_feats)
    return top_feats

# %% [markdown]
# ## 8. Model Training & Evaluation with Hyperparameter Tuning (7 Models)  

# %%
def train_and_evaluate_all_models(X_train, X_test, y_train, y_test):
    """Train and evaluate 7 different ML models with hyperparameter tuning."""
    
    models_results = {}
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATING 7 MACHINE LEARNING MODELS")
    print("="*60)
    
    # 1. Logistic Regression
    print("\n[1/7] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_param_dist = {
        'C': uniform(0.01, 10.0),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced']
    }
    lr_search = RandomizedSearchCV(
        lr, lr_param_dist,
        n_iter=20, cv=5, random_state=42, n_jobs=-1, verbose=0
    )
    lr_search.fit(X_train, y_train)
    models_results['Logistic Regression'] = lr_search.best_estimator_
    print(f"Best params: {lr_search.best_params_}")
    evaluate_classifier(lr_search.best_estimator_, X_test, y_test, "Logistic Regression")

    # 2. Decision Tree
    print("\n[2/7] Training Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt_param_dist = {
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    }
    dt_search = RandomizedSearchCV(
        dt, dt_param_dist,
        n_iter=20, cv=5, random_state=42, n_jobs=-1, verbose=0
    )
    dt_search.fit(X_train, y_train)
    models_results['Decision Tree'] = dt_search.best_estimator_
    print(f"Best params: {dt_search.best_params_}")
    evaluate_classifier(dt_search.best_estimator_, X_test, y_test, "Decision Tree")

    # 3. Random Forest
    print("\n[3/7] Training Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    rf_param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'class_weight': [None, 'balanced']
    }
    rf_search = RandomizedSearchCV(
        rf, rf_param_dist,
        n_iter=20, cv=5, random_state=42, n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train, y_train)
    models_results['Random Forest'] = rf_search.best_estimator_
    print(f"Best params: {rf_search.best_params_}")
    evaluate_classifier(rf_search.best_estimator_, X_test, y_test, "Random Forest")

    # 4. Support Vector Machine
    print("\n[4/7] Training Support Vector Machine...")
    svm = SVC(probability=True, random_state=42)
    svm_param_dist = {
        'C': uniform(0.1, 10.0),
        'gamma': ['scale', 'auto'] + list(uniform(0.001, 1.0).rvs(5)),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'class_weight': [None, 'balanced']
    }
    svm_search = RandomizedSearchCV(
        svm, svm_param_dist,
        n_iter=20, cv=5, random_state=42, n_jobs=-1, verbose=0
    )
    svm_search.fit(X_train, y_train)
    models_results['SVM'] = svm_search.best_estimator_
    print(f"Best params: {svm_search.best_params_}")
    evaluate_classifier(svm_search.best_estimator_, X_test, y_test, "Support Vector Machine")

    # 5. K-Nearest Neighbors
    print("\n[5/7] Training K-Nearest Neighbors...")
    knn = KNeighborsClassifier()
    knn_param_dist = {
        'n_neighbors': randint(3, 15),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
    }
    knn_search = RandomizedSearchCV(
        knn, knn_param_dist,
        n_iter=20, cv=5, random_state=42, n_jobs=-1, verbose=0
    )
    knn_search.fit(X_train, y_train)
    models_results['KNN'] = knn_search.best_estimator_
    print(f"Best params: {knn_search.best_params_}")
    evaluate_classifier(knn_search.best_estimator_, X_test, y_test, "K-Nearest Neighbors")

    # 6. Naive Bayes
    print("\n[6/7] Training Naive Bayes...")
    nb = GaussianNB()
    nb_param_dist = {
        'var_smoothing': uniform(1e-10, 1e-8)
    }
    nb_search = RandomizedSearchCV(
        nb, nb_param_dist,
        n_iter=10, cv=5, random_state=42, n_jobs=-1, verbose=0
    )
    nb_search.fit(X_train, y_train)
    models_results['Naive Bayes'] = nb_search.best_estimator_
    print(f"Best params: {nb_search.best_params_}")
    evaluate_classifier(nb_search.best_estimator_, X_test, y_test, "Naive Bayes")

    # 7. Gradient Boosting
    print("\n[7/7] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(random_state=42)
    gb_param_dist = {
        'n_estimators': randint(50, 200),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
    }
    gb_search = RandomizedSearchCV(
        gb, gb_param_dist,
        n_iter=20, cv=5, random_state=42, n_jobs=-1, verbose=0
    )
    gb_search.fit(X_train, y_train)
    models_results['Gradient Boosting'] = gb_search.best_estimator_
    print(f"Best params: {gb_search.best_params_}")
    evaluate_classifier(gb_search.best_estimator_, X_test, y_test, "Gradient Boosting")

    # Model Comparison Summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    accuracies = {}
    for name, model in models_results.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy
        print(f"{name:20s}: {accuracy:.4f}")
    
    # Plot model comparison
    plt.figure(figsize=(12, 8))
    models = list(accuracies.keys())
    scores = list(accuracies.values())
    
    bars = plt.bar(models, scores, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Model Accuracy Comparison (Asia vs Europe Prediction)', fontsize=14, fontweight='bold')
    plt.xlabel('Machine Learning Models', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    # Find best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model_name]
    print(f"\n🏆 BEST MODEL: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    return models_results, accuracies

# %% [markdown]
# ## 9. Main Execution  

# %%
def predict_continent(model, scaler, feature_values, feature_names, continent_encoder):
    """Make prediction for new data point."""
    # Create DataFrame with feature names
    new_data = pd.DataFrame([feature_values], columns=feature_names)
    
    # Scale the features
    new_data_scaled = scaler.transform(new_data)
    
    # Make prediction
    prediction_encoded = model.predict(new_data_scaled)[0]
    prediction_proba = model.predict_proba(new_data_scaled)[0]
    
    # Decode prediction (assuming 0=Asia, 1=Europe based on typical label encoding)
    continent_names = ['Asia', 'Europe']  # This might need adjustment based on actual encoding
    predicted_continent = continent_names[prediction_encoded]
    confidence = max(prediction_proba)
    
    return predicted_continent, confidence, prediction_proba

def main():
    # 9.1 Load & Preprocess
    df = load_data('Co2_Emissions_by_Sectors_Europe-Asia.csv')
    features = [
        'Co2_Emissions_MetricTons', 'Energy_Consumption_TWh',
        'Automobile_Co2_Emissions_MetricTons',
        'Industrial_Co2_Emissions_MetricTons',
        'Agriculture_Co2_Emissions_MetricTons',
        'Domestic_Co2_Emissions_MetricTons',
        'Population_Millions', 'GDP_Billion_USD',
        'Urbanization_Percentage', 'Renewable_Energy_Percentage',
        'Industrial_Growth_Percentage', 'Transport_Growth_Percentage',
        'Industry_Type_Encoded'
    ]
    X_train, X_test, y_train, y_test = preprocess(df, features)

    # 9.2 Exploratory Data Analysis
    exploratory_analysis(df, features)

    # 9.3 Clustering (for insight)
    cluster_analysis(np.vstack((X_train, X_test)))

    # 9.4 Feature Selection (Top 5)
    top5 = select_top_features(X_train, y_train, features, top_n=5)

    # 9.5 Train & Evaluate all 7 models using Top 5 features
    X_top = StandardScaler().fit_transform(df[top5].values)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_top, df['Continent_Encoded'].values,
        test_size=0.2, random_state=42, stratify=df['Continent_Encoded']
    )
    
    # Store the scaler for later predictions
    scaler = StandardScaler()
    X_top_scaled = scaler.fit_transform(df[top5].values)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_top_scaled, df['Continent_Encoded'].values,
        test_size=0.2, random_state=42, stratify=df['Continent_Encoded']
    )
    
    models_results, accuracies = train_and_evaluate_all_models(X_tr, X_te, y_tr, y_te)
    
    # 9.6 Get the best model and export it
    print("\n" + "="*60)
    print("EXPORTING BEST MODEL")
    print("="*60)
    
    # Get the best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models_results[best_model_name]
    best_accuracy = accuracies[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📊 Accuracy: {best_accuracy:.4f}")
    
    # Export the model
    model_filename = 'best_model.pkl'
    pickle.dump(best_model, open(model_filename, 'wb'))
    print(f"\n✅ Model exported to: {model_filename}")
    
    # Export the scaler as well (needed for predictions)
    scaler_filename = 'scaler.pkl'
    pickle.dump(scaler, open(scaler_filename, 'wb'))
    print(f"✅ Scaler exported to: {scaler_filename}")
    
    # Export feature names
    feature_info = {
        'feature_names': top5,
        'model_name': best_model_name,
        'accuracy': best_accuracy
    }
    feature_filename = 'model_info.pkl'
    pickle.dump(feature_info, open(feature_filename, 'wb'))
    print(f"✅ Model info exported to: {feature_filename}")
    
    print("\n=== Analysis Complete ===")
    print("🎯 The best model has been exported and can be used for predictions")
    print("📊 Based on CO2 emissions and related environmental/economic factors")

if __name__ == "__main__":
    main()


