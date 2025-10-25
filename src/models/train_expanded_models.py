"""
Train ML models with expanded dataset (1,483 records)
Compare with baseline (821 records)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, f1_score, roc_auc_score, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info("TRAINING MODELS WITH EXPANDED DATASET (1,483 RECORDS)")
logger.info("="*70)

# Load expanded ML-ready data
df = pd.read_csv('data/processed/transfers_ml_ready_expanded.csv')
logger.info(f"\nLoaded {len(df)} records with {len(df.columns)} features")

# Select features for modeling
feature_cols = [
    'perf_before_goals', 'perf_before_assists', 'perf_before_minutes',
    'goals_per_90_before', 'assists_per_90_before',
    'goal_contribution_before',
    'age', 'is_young', 'is_prime', 'is_veteran',
    'is_forward', 'is_midfielder', 'is_defender', 'is_goalkeeper',
    'fee_millions'
]

# Add league dummies
league_cols = [c for c in df.columns if c.startswith('league_')]
feature_cols.extend(league_cols)

# Filter to valid rows
df_model = df[feature_cols + ['target_goals_after', 'target_success_goals']].dropna()
logger.info(f"\nValid records for modeling: {len(df_model)}")

X = df_model[feature_cols]
y_reg = df_model['target_goals_after']
y_clf = df_model['target_success_goals']

# Train-test split
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

logger.info(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Results storage
results = {
    'dataset_size': len(df_model),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'n_features': len(feature_cols),
    'models': {}
}

logger.info("\n" + "="*70)
logger.info("CLASSIFICATION: Goal Improvement Prediction")
logger.info("="*70)

# Random Forest Classifier
logger.info("\nTraining Random Forest Classifier...")
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train, y_clf_train)
y_pred_rf = rf_clf.predict(X_test)
y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]

rf_f1 = f1_score(y_clf_test, y_pred_rf)
rf_auc = roc_auc_score(y_clf_test, y_proba_rf)
logger.info(f"  F1-Score: {rf_f1:.4f}")
logger.info(f"  ROC-AUC: {rf_auc:.4f}")

results['models']['random_forest_clf'] = {
    'f1_score': float(rf_f1),
    'roc_auc': float(rf_auc)
}

# XGBoost Classifier
logger.info("\nTraining XGBoost Classifier...")
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_clf.fit(X_train, y_clf_train)
y_pred_xgb = xgb_clf.predict(X_test)
y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]

xgb_f1 = f1_score(y_clf_test, y_pred_xgb)
xgb_auc = roc_auc_score(y_clf_test, y_proba_xgb)
logger.info(f"  F1-Score: {xgb_f1:.4f}")
logger.info(f"  ROC-AUC: {xgb_auc:.4f}")

results['models']['xgboost_clf'] = {
    'f1_score': float(xgb_f1),
    'roc_auc': float(xgb_auc)
}

# LightGBM Classifier
logger.info("\nTraining LightGBM Classifier...")
lgb_clf = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
lgb_clf.fit(X_train, y_clf_train)
y_pred_lgb = lgb_clf.predict(X_test)
y_proba_lgb = lgb_clf.predict_proba(X_test)[:, 1]

lgb_f1 = f1_score(y_clf_test, y_pred_lgb)
lgb_auc = roc_auc_score(y_clf_test, y_proba_lgb)
logger.info(f"  F1-Score: {lgb_f1:.4f}")
logger.info(f"  ROC-AUC: {lgb_auc:.4f}")

results['models']['lightgbm_clf'] = {
    'f1_score': float(lgb_f1),
    'roc_auc': float(lgb_auc)
}

logger.info("\n" + "="*70)
logger.info("REGRESSION: Goals After Transfer")
logger.info("="*70)

# Random Forest Regressor
logger.info("\nTraining Random Forest Regressor...")
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_reg.fit(X_train, y_reg_train)
y_pred_rf_reg = rf_reg.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_rf_reg))
rf_r2 = r2_score(y_reg_test, y_pred_rf_reg)
logger.info(f"  RMSE: {rf_rmse:.4f}")
logger.info(f"  R²: {rf_r2:.4f}")

results['models']['random_forest_reg'] = {
    'rmse': float(rf_rmse),
    'r2': float(rf_r2)
}

# XGBoost Regressor
logger.info("\nTraining XGBoost Regressor...")
xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_reg.fit(X_train, y_reg_train)
y_pred_xgb_reg = xgb_reg.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_xgb_reg))
xgb_r2 = r2_score(y_reg_test, y_pred_xgb_reg)
logger.info(f"  RMSE: {xgb_rmse:.4f}")
logger.info(f"  R²: {xgb_r2:.4f}")

results['models']['xgboost_reg'] = {
    'rmse': float(xgb_rmse),
    'r2': float(xgb_r2)
}

# LightGBM Regressor
logger.info("\nTraining LightGBM Regressor...")
lgb_reg = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
lgb_reg.fit(X_train, y_reg_train)
y_pred_lgb_reg = lgb_reg.predict(X_test)

lgb_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_lgb_reg))
lgb_r2 = r2_score(y_reg_test, y_pred_lgb_reg)
logger.info(f"  RMSE: {lgb_rmse:.4f}")
logger.info(f"  R²: {lgb_r2:.4f}")

results['models']['lightgbm_reg'] = {
    'rmse': float(lgb_rmse),
    'r2': float(lgb_r2)
}

# Save models
logger.info("\n" + "="*70)
logger.info("SAVING MODELS")
logger.info("="*70)

joblib.dump(xgb_clf, 'results/models/xgboost_clf_expanded.pkl')
joblib.dump(xgb_reg, 'results/models/xgboost_reg_expanded.pkl')
logger.info("\n✅ Models saved")

# Save results
with open('results/expanded_model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
logger.info("✅ Results saved")

# Summary
logger.info("\n" + "="*70)
logger.info("SUMMARY")
logger.info("="*70)
logger.info(f"\nDataset: {len(df_model)} records")
logger.info(f"Features: {len(feature_cols)}")
logger.info(f"\nBest Classification: LightGBM (F1={lgb_f1:.4f}, AUC={lgb_auc:.4f})")
logger.info(f"Best Regression: XGBoost (RMSE={xgb_rmse:.4f}, R²={xgb_r2:.4f})")

logger.info("\n" + "="*70)
logger.info("TRAINING COMPLETE!")
logger.info("="*70)

