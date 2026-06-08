from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

california_housing = fetch_california_housing()
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
print("First 5 rows of the California housing dataset:")
print(data.head())
print(data.shape)
print(data.describe())
X = data.values
y = california_housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

#======================Linear Regression======================
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

#======================Ridge Regression======================
ridge_model = Ridge(alpha=0.8)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
#======================Random Forest Regressor======================
base_rf = RandomForestRegressor(max_depth=8, random_state=42)
ada_rf = AdaBoostRegressor(estimator=base_rf, n_estimators=50, random_state=42)
ada_rf.fit(X_train, y_train)
y_pred_ada_rf = ada_rf.predict(X_test)


#======================Evaluation======================
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{model_name} - Mean Squared Error: {mse:.4f}, R^2 Score: {r2:.4f}, Mean Absolute Error: {mae:.4f}, RMSE: {rmse:.4f}")

evaluate_model(y_test, y_pred_linear, "Linear Regression")
evaluate_model(y_test, y_pred_ridge, "Ridge Regression")    
evaluate_model(y_test, y_pred_ada_rf, "AdaBoost Random Forest Regressor")


#======================Visualization and Importance======================
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

# 单独训练一个随机森林用于对比特征重要性
rf_importance = RandomForestRegressor(max_depth=8, random_state=42)
rf_importance.fit(X_train, y_train)

feature_names = np.array(california_housing.feature_names)

# 画出随机森林的特征重要性
plot_feature_importances(rf_importance.feature_importances_,
                         'Random Forest Feature Importances',
                         feature_names)

# 计算 AdaBoost 中所有基学习器的平均特征重要性
ada_importances = np.mean([est.feature_importances_ for est in ada_rf.estimators_], axis=0)

# 画出 AdaBoost 的平均特征重要性
plot_feature_importances(ada_importances,
                         'AdaBoost Random Forest Feature Importances',
                         feature_names)

# 打印特征重要性数值
print("\nFeature Importances (Random Forest):")
print(pd.DataFrame({'Feature': feature_names, 'Importance': rf_importance.feature_importances_})
      .sort_values(by='Importance', ascending=False))
print("\nFeature Importances (AdaBoost RF average):")
print(pd.DataFrame({'Feature': feature_names, 'Importance': ada_importances})
      .sort_values(by='Importance', ascending=False))