import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. LOAD
df = pd.read_csv('/mnt/user-data/uploads/IMDb_Movies_India.csv', encoding='latin1')

# 2. PREPROCESSING
df = df.dropna(subset=['Rating'])

df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Year'].fillna(df['Year'].median(), inplace=True)

df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
df['Duration'].fillna(df['Duration'].median(), inplace=True)

df['Votes'] = df['Votes'].astype(str).str.replace(',', '').str.strip()
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Votes'].fillna(df['Votes'].median(), inplace=True)

for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col].fillna('Unknown', inplace=True)

# 3. FEATURE ENGINEERING
df['Genre_Primary'] = df['Genre'].str.split(',').str[0].str.strip()
df['Votes_log'] = np.log1p(df['Votes'])

for col in ['Genre_Primary', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

FEATURES = ['Year', 'Duration', 'Genre_Primary_enc', 'Director_enc',
            'Actor 1_enc', 'Actor 2_enc', 'Actor 3_enc', 'Votes_log']
TARGET = 'Rating'

df_model = df[FEATURES + [TARGET, 'Name']].dropna()
print(f"Clean modeling dataset: {df_model.shape}")

X = df_model[FEATURES]
y = df_model[TARGET]
names = df_model['Name']

print(f"Rating stats:\n{y.describe()}\n")

# 4. SPLIT
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X, y, names, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 5. MODELS
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {'MAE': round(mae,3), 'RMSE': round(rmse,3), 'R2': round(r2,3), 'preds': preds}
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

best = max(results, key=lambda k: results[k]['R2'])
print(f"\nBest model: {best} (R2={results[best]['R2']})")

# 6. PLOTS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Movie Rating Prediction - IMDb India Dataset', fontsize=15, fontweight='bold')

# Rating distribution
axes[0,0].hist(df['Rating'], bins=30, color='steelblue', edgecolor='white', alpha=0.85)
axes[0,0].axvline(df['Rating'].mean(), color='red', linestyle='--', label=f"Mean: {df['Rating'].mean():.2f}")
axes[0,0].set_title('Rating Distribution')
axes[0,0].set_xlabel('IMDb Rating')
axes[0,0].set_ylabel('Count')
axes[0,0].legend()

# Model comparison
mnames = list(results.keys())
r2s = [results[m]['R2'] for m in mnames]
maes = [results[m]['MAE'] for m in mnames]
x = np.arange(len(mnames))
w = 0.35
b1 = axes[0,1].bar(x - w/2, r2s, w, label='R2 Score', color='steelblue', alpha=0.85)
axes[0,1].bar(x + w/2, maes, w, label='MAE', color='coral', alpha=0.85)
axes[0,1].set_title('Model Comparison')
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(['Lin. Reg.', 'Rand. Forest', 'Grad. Boost'], fontsize=8)
axes[0,1].legend()
for bar in b1:
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                   f'{bar.get_height():.2f}', ha='center', fontsize=9)

# Actual vs Predicted
axes[1,0].scatter(y_test, results[best]['preds'], alpha=0.3, color='steelblue', s=10)
mn, mx = y_test.min(), y_test.max()
axes[1,0].plot([mn,mx],[mn,mx],'r--',lw=2)
axes[1,0].set_title(f'Actual vs Predicted ({best})')
axes[1,0].set_xlabel('Actual Rating')
axes[1,0].set_ylabel('Predicted Rating')

# Feature importance
rf = models['Random Forest']
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
imp.plot(kind='barh', ax=axes[1,1], color='steelblue', alpha=0.85)
axes[1,1].set_title('Feature Importance (Random Forest)')
axes[1,1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/movie_rating_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved.")

# Sample predictions
print("\n--- Sample Predictions ---")
sample = pd.DataFrame({
    'Movie': names_test.values[:10],
    'Actual': y_test.values[:10],
    'Predicted (RF)': results['Random Forest']['preds'][:10].round(2)
})
print(sample.to_string(index=False))

