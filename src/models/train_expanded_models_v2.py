"""
Train ML models with fixed expanded dataset (1,483 records, 82 features)
Using same feature set as baseline for fair comparison
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.metrics import classification_report, f1_score, roc_auc_score, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("TRAINING MODELS WITH FIXED EXPANDED DATASET")
logger.info("="*80)

# Load fixed expanded dataset
df = pd.read_csv('data/processed/transfers_ml_ready_expanded_fixed.csv')
logger.info(f"\nLoaded {len(df)} records with {len(df.columns)} features")

# Select comprehensive feature set (matching baseline approach)
feature_cols = [
    # Performance metrics (before transfer)
    'perf_before_goals', 'perf_before_assists', 'perf_before_minutes',
    'goals_per_90_before', 'assists_per_90_before', 'goal_contribution_before',
    
    # Player attributes
    'age', 'is_young', 'is_prime', 'is_veteran',
    'is_forward', 'is_midfielder', 'is_defender', 'is_goalkeeper',
    
    # Transfer details
    'fee_millions', 'fee_log', 'has_fee',
    
    # Comparative metrics (vs league/position averages)
    'goals_vs_league_avg', 'assists_vs_league_avg',
    'goals_vs_position_avg', 'assists_vs_position_avg',
    
    # Performance changes
    'goals_change', 'assists_change', 'goal_contribution_change',
    'minutes_change', 'minutes_per_match_before', 'minutes_per_match_after'
]

# Add league dummies
league_cols = [c for c in df.columns if c.startswith('league_')]
feature_cols.extend(league_cols)

# Filter available features
available_features = [f for f in feature_cols if f in df.columns]
logger.info(f"\nUsing {len(available_features)} features for modeling")

# Prepare data
target_cols = ['target_goals_after', 'target_success_goals']
df_model = df[available_features + target_cols].copy()

# Ensure all features are numeric
for col in available_features:
    if df_model[col].dtype == 'object':
        logger.warning(f"Removing non-numeric column: {col}")
        available_features.remove(col)

# Re-select only numeric features
df_model = df[available_features + target_cols].copy()

# Handle missing values
df_model = df_model.fillna(0)
logger.info(f"Valid records for modeling: {len(df_model)}")

X = df_model[available_features]
y_reg = df_model['target_goals_after']
y_clf = df_model['target_success_goals']

# Train-test split (same random state as baseline for consistency)
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

logger.info(f"\nTrain size: {len(X_train)} ({len(X_train)/len(df_model)*100:.1f}%)")
logger.info(f"Test size: {len(X_test)} ({len(X_test)/len(df_model)*100:.1f}%)")
logger.info(f"Features: {len(available_features)}")

# Results storage
results = {
    'dataset_size': len(df_model),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'n_features': len(available_features),
    'models': {}
}

# Create results directory
Path('results/expanded_v2').mkdir(parents=True, exist_ok=True)
Path('models/expanded_v2').mkdir(parents=True, exist_ok=True)

# ============================================================================
# CLASSIFICATION MODELS
# ============================================================================
logger.info("\n" + "="*80)
logger.info("CLASSIFICATION: Goal Improvement Prediction")
logger.info("="*80)

# Random Forest Classifier
logger.info("\n1. Training Random Forest Classifier...")
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, 
                                min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_clf_train)
y_pred_rf_clf = rf_clf.predict(X_test)
y_proba_rf_clf = rf_clf.predict_proba(X_test)[:, 1]

rf_clf_metrics = {
    'accuracy': accuracy_score(y_clf_test, y_pred_rf_clf),
    'precision': precision_score(y_clf_test, y_pred_rf_clf, zero_division=0),
    'recall': recall_score(y_clf_test, y_pred_rf_clf, zero_division=0),
    'f1_score': f1_score(y_clf_test, y_pred_rf_clf, zero_division=0),
    'roc_auc': roc_auc_score(y_clf_test, y_proba_rf_clf)
}
results['models']['random_forest_clf'] = rf_clf_metrics
logger.info(f"   Accuracy: {rf_clf_metrics['accuracy']:.4f}")
logger.info(f"   F1-Score: {rf_clf_metrics['f1_score']:.4f}")
logger.info(f"   ROC-AUC: {rf_clf_metrics['roc_auc']:.4f}")

joblib.dump(rf_clf, 'models/expanded_v2/random_forest_clf.pkl')

# XGBoost Classifier
logger.info("\n2. Training XGBoost Classifier...")
xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb_clf.fit(X_train, y_clf_train)
y_pred_xgb_clf = xgb_clf.predict(X_test)
y_proba_xgb_clf = xgb_clf.predict_proba(X_test)[:, 1]

xgb_clf_metrics = {
    'accuracy': accuracy_score(y_clf_test, y_pred_xgb_clf),
    'precision': precision_score(y_clf_test, y_pred_xgb_clf, zero_division=0),
    'recall': recall_score(y_clf_test, y_pred_xgb_clf, zero_division=0),
    'f1_score': f1_score(y_clf_test, y_pred_xgb_clf, zero_division=0),
    'roc_auc': roc_auc_score(y_clf_test, y_proba_xgb_clf)
}
results['models']['xgboost_clf'] = xgb_clf_metrics
logger.info(f"   Accuracy: {xgb_clf_metrics['accuracy']:.4f}")
logger.info(f"   F1-Score: {xgb_clf_metrics['f1_score']:.4f}")
logger.info(f"   ROC-AUC: {xgb_clf_metrics['roc_auc']:.4f}")

joblib.dump(xgb_clf, 'models/expanded_v2/xgboost_clf.pkl')

# LightGBM Classifier
logger.info("\n3. Training LightGBM Classifier...")
lgb_clf = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=-1, verbose=-1)
lgb_clf.fit(X_train, y_clf_train)
y_pred_lgb_clf = lgb_clf.predict(X_test)
y_proba_lgb_clf = lgb_clf.predict_proba(X_test)[:, 1]

lgb_clf_metrics = {
    'accuracy': accuracy_score(y_clf_test, y_pred_lgb_clf),
    'precision': precision_score(y_clf_test, y_pred_lgb_clf, zero_division=0),
    'recall': recall_score(y_clf_test, y_pred_lgb_clf, zero_division=0),
    'f1_score': f1_score(y_clf_test, y_pred_lgb_clf, zero_division=0),
    'roc_auc': roc_auc_score(y_clf_test, y_proba_lgb_clf)
}
results['models']['lightgbm_clf'] = lgb_clf_metrics
logger.info(f"   Accuracy: {lgb_clf_metrics['accuracy']:.4f}")
logger.info(f"   F1-Score: {lgb_clf_metrics['f1_score']:.4f}")
logger.info(f"   ROC-AUC: {lgb_clf_metrics['roc_auc']:.4f}")

joblib.dump(lgb_clf, 'models/expanded_v2/lightgbm_clf.pkl')

# Voting Classifier
logger.info("\n4. Training Voting Classifier (Ensemble)...")
voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('xgb', xgb_clf), ('lgb', lgb_clf)],
    voting='soft'
)
voting_clf.fit(X_train, y_clf_train)
y_pred_voting_clf = voting_clf.predict(X_test)
y_proba_voting_clf = voting_clf.predict_proba(X_test)[:, 1]

voting_clf_metrics = {
    'accuracy': accuracy_score(y_clf_test, y_pred_voting_clf),
    'precision': precision_score(y_clf_test, y_pred_voting_clf, zero_division=0),
    'recall': recall_score(y_clf_test, y_pred_voting_clf, zero_division=0),
    'f1_score': f1_score(y_clf_test, y_pred_voting_clf, zero_division=0),
    'roc_auc': roc_auc_score(y_clf_test, y_proba_voting_clf)
}
results['models']['voting_classifier'] = voting_clf_metrics
logger.info(f"   Accuracy: {voting_clf_metrics['accuracy']:.4f}")
logger.info(f"   F1-Score: {voting_clf_metrics['f1_score']:.4f}")
logger.info(f"   ROC-AUC: {voting_clf_metrics['roc_auc']:.4f}")

joblib.dump(voting_clf, 'models/expanded_v2/voting_classifier.pkl')

# ============================================================================
# REGRESSION MODELS
# ============================================================================
logger.info("\n" + "="*80)
logger.info("REGRESSION: Goals After Transfer Prediction")
logger.info("="*80)

# Random Forest Regressor
logger.info("\n1. Training Random Forest Regressor...")
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                               min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_reg_train)
y_pred_rf_reg = rf_reg.predict(X_test)

rf_reg_metrics = {
    'rmse': np.sqrt(mean_squared_error(y_reg_test, y_pred_rf_reg)),
    'mae': np.mean(np.abs(y_reg_test - y_pred_rf_reg)),
    'r2': r2_score(y_reg_test, y_pred_rf_reg)
}
results['models']['random_forest_reg'] = rf_reg_metrics
logger.info(f"   RMSE: {rf_reg_metrics['rmse']:.4f}")
logger.info(f"   MAE: {rf_reg_metrics['mae']:.4f}")
logger.info(f"   R²: {rf_reg_metrics['r2']:.4f}")

joblib.dump(rf_reg, 'models/expanded_v2/random_forest_reg.pkl')

# XGBoost Regressor
logger.info("\n2. Training XGBoost Regressor...")
xgb_reg = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb_reg.fit(X_train, y_reg_train)
y_pred_xgb_reg = xgb_reg.predict(X_test)

xgb_reg_metrics = {
    'rmse': np.sqrt(mean_squared_error(y_reg_test, y_pred_xgb_reg)),
    'mae': np.mean(np.abs(y_reg_test - y_pred_xgb_reg)),
    'r2': r2_score(y_reg_test, y_pred_xgb_reg)
}
results['models']['xgboost_reg'] = xgb_reg_metrics
logger.info(f"   RMSE: {xgb_reg_metrics['rmse']:.4f}")
logger.info(f"   MAE: {xgb_reg_metrics['mae']:.4f}")
logger.info(f"   R²: {xgb_reg_metrics['r2']:.4f}")

joblib.dump(xgb_reg, 'models/expanded_v2/xgboost_reg.pkl')

# LightGBM Regressor
logger.info("\n3. Training LightGBM Regressor...")
lgb_reg = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                           num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                           random_state=42, n_jobs=-1, verbose=-1)
lgb_reg.fit(X_train, y_reg_train)
y_pred_lgb_reg = lgb_reg.predict(X_test)

lgb_reg_metrics = {
    'rmse': np.sqrt(mean_squared_error(y_reg_test, y_pred_lgb_reg)),
    'mae': np.mean(np.abs(y_reg_test - y_pred_lgb_reg)),
    'r2': r2_score(y_reg_test, y_pred_lgb_reg)
}
results['models']['lightgbm_reg'] = lgb_reg_metrics
logger.info(f"   RMSE: {lgb_reg_metrics['rmse']:.4f}")
logger.info(f"   MAE: {lgb_reg_metrics['mae']:.4f}")
logger.info(f"   R²: {lgb_reg_metrics['r2']:.4f}")

joblib.dump(lgb_reg, 'models/expanded_v2/lightgbm_reg.pkl')

# Voting Regressor
logger.info("\n4. Training Voting Regressor (Ensemble)...")
voting_reg = VotingRegressor(
    estimators=[('rf', rf_reg), ('xgb', xgb_reg), ('lgb', lgb_reg)]
)
voting_reg.fit(X_train, y_reg_train)
y_pred_voting_reg = voting_reg.predict(X_test)

voting_reg_metrics = {
    'rmse': np.sqrt(mean_squared_error(y_reg_test, y_pred_voting_reg)),
    'mae': np.mean(np.abs(y_reg_test - y_pred_voting_reg)),
    'r2': r2_score(y_reg_test, y_pred_voting_reg)
}
results['models']['voting_regressor'] = voting_reg_metrics
logger.info(f"   RMSE: {voting_reg_metrics['rmse']:.4f}")
logger.info(f"   MAE: {voting_reg_metrics['mae']:.4f}")
logger.info(f"   R²: {voting_reg_metrics['r2']:.4f}")

joblib.dump(voting_reg, 'models/expanded_v2/voting_regressor.pkl')

# Save results
with open('results/expanded_v2/model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

logger.info("\n" + "="*80)
logger.info("TRAINING COMPLETE!")
logger.info("="*80)
logger.info(f"\n✅ Models saved to: models/expanded_v2/")
logger.info(f"✅ Results saved to: results/expanded_v2/model_results.json")

# Print summary
logger.info("\n" + "="*80)
logger.info("PERFORMANCE SUMMARY")
logger.info("="*80)

logger.info("\nBest Classification Model:")
best_clf = max([(k, v['f1_score']) for k, v in results['models'].items() if 'clf' in k or 'classifier' in k], key=lambda x: x[1])
logger.info(f"  {best_clf[0]}: F1 = {best_clf[1]:.4f}")

logger.info("\nBest Regression Model:")
best_reg = max([(k, v['r2']) for k, v in results['models'].items() if 'reg' in k or 'regressor' in k], key=lambda x: x[1])
logger.info(f"  {best_reg[0]}: R² = {best_reg[1]:.4f}")

logger.info("\n" + "="*80)

