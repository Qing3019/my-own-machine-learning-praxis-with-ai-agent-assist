import csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

def load_csv_dataset(file_path):
    csv_reader = csv.reader(open(file_path, 'r'), delimiter=',')
    x, y = [], []
    for row in csv_reader:
        x.append(row[2:13])
        y.append(row[-1])

    feature_names = np.array(x[0])
    return np.array(x[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names

def plot_feature_importances(feature_importances, title, feature_names):
    # 将重要性值标准化到 0~100 范围
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    # 将得分从高到低排序
    index_sorted = np.flipud(np.argsort(feature_importances))
    # 让 X 坐标轴上的标签居中显示
    pos = np.arange(index_sorted.shape[0]) + 0.5
    # 画条形图
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title.replace(' ', '_').lower() + '.png', dpi=150, bbox_inches='tight')
    print(f"Chart saved: {title}")

def plot_predictions(y_true, y_pred, title):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title.replace(' ', '_').lower() + '.png', dpi=150, bbox_inches='tight')
    print(f"Chart saved: {title}")

def plot_actual_vs_predicted_line(y_true, y_pred, title, num_samples=50):
    n = min(num_samples, len(y_true))
    x = range(n)
    plt.figure(figsize=(12, 6))
    plt.plot(x, y_true[:n], 'b-', alpha=0.6, label='Actual')
    plt.plot(x, y_pred[:n], 'r--', alpha=0.6, label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Bike Rentals (cnt)')
    plt.title(title + f' (first {n} samples)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(title.replace(' ', '_').lower() + '.png', dpi=150, bbox_inches='tight')
    print(f"Chart saved: {title}")

file_path = 'day.csv'
X, y, feature_names = load_csv_dataset(file_path)
X, y = shuffle(X, y, random_state=7)
print("First 5 rows of the dataset:")
print(X[:5])
print("Shape of the dataset:", X.shape)

num_training = int(0.9 * len(X))
X_train, y_train, X_test, y_test = X[:num_training], y[:num_training], X[num_training:], y[num_training:]

rf_regressor = RandomForestRegressor(n_estimators=1000,max_depth=10,min_samples_split=5, random_state=42)
rf_regressor.fit(X_train, y_train)
y_pred_rf = rf_regressor.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
evs_rf = explained_variance_score(y_test, y_pred_rf)
print(f"Random Forest Regressor - Mean Squared Error: {mse_rf:.4f}, R^2 Score: {r2_rf:.4f}, Mean Absolute Error: {mae_rf:.4f}, RMSE: {rmse_rf:.4f}, Explained Variance Score: {evs_rf:.4f}")

feature_importances_rf = rf_regressor.feature_importances_
plot_feature_importances(feature_importances_rf, "Random Forest Regressor Feature Importances", feature_names)
plot_predictions(y_test, y_pred_rf, "Random Forest Regressor Predictions")
plot_actual_vs_predicted_line(y_test, y_pred_rf, "Random Forest Regressor Actual vs Predicted")