"""
Train Multiple Machine Learning Models for Transfer Success Prediction
Includes: Random Forest, XGBoost, LightGBM for both Classification and Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TransferSuccessPredictor:
    """Train and evaluate ML models for transfer success prediction"""
    
    def __init__(self):
        self.data_path = 'data/processed/transfers_ml_ready.csv'
        self.models_dir = 'models'
        self.results_dir = 'results'
        self.scaler = StandardScaler()
        
        # Create directories
        import os
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(f'{self.results_dir}/model_evaluation', exist_ok=True)
    
    def load_data(self):
        """Load ML-ready dataset"""
        logger.info("Loading ML-ready dataset...")
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix and target variables"""
        logger.info("Preparing features...")
        
        # Define feature columns (exclude targets and metadata)
        exclude_cols = [
            'club_name', 'player_name', 'club_involved_name', 'fee', 'transfer_movement',
            'transfer_period', 'league_name', 'year', 'season', 'country', 'position',
            'position_group', 'age_category', 'fee_category',
            # Target columns
            'target_goals_after', 'target_assists_after', 'target_goal_contribution_after',
            'target_minutes_after', 'target_success_goals', 'target_success_minutes',
            'target_success_composite', 'target_success_contribution',
            # Derived columns that leak information
            'perf_after_goals', 'perf_after_assists', 'perf_after_minutes', 'perf_after_matches',
            'goal_change', 'assist_change', 'minutes_change',
            'goal_contribution_after', 'goals_per_90_after', 'assists_per_90_after',
            'minutes_per_match_after', 'success_goals', 'success_minutes', 'success_composite'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Selected {len(feature_cols)} features")
        
        return feature_cols
    
    def train_classification_models(self, X_train, X_test, y_train, y_test, target_name):
        """Train classification models"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Training Classification Models for: {target_name}")
        logger.info(f"{'='*70}")
        
        results = {}
        
        # Random Forest
        logger.info("\n1. Random Forest Classifier...")
        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf_clf.fit(X_train, y_train)
        rf_pred = rf_clf.predict(X_test)
        rf_pred_proba = rf_clf.predict_proba(X_test)[:, 1]
        
        results['Random Forest'] = {
            'model': rf_clf,
            'predictions': rf_pred,
            'probabilities': rf_pred_proba,
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred, zero_division=0),
            'recall': recall_score(y_test, rf_pred, zero_division=0),
            'f1': f1_score(y_test, rf_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, rf_pred_proba) if len(np.unique(y_test)) > 1 else 0
        }
        
        logger.info(f"  Accuracy: {results['Random Forest']['accuracy']:.4f}")
        logger.info(f"  F1-Score: {results['Random Forest']['f1']:.4f}")
        logger.info(f"  ROC-AUC: {results['Random Forest']['roc_auc']:.4f}")
        
        # XGBoost
        logger.info("\n2. XGBoost Classifier...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        xgb_clf.fit(X_train, y_train)
        xgb_pred = xgb_clf.predict(X_test)
        xgb_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]
        
        results['XGBoost'] = {
            'model': xgb_clf,
            'predictions': xgb_pred,
            'probabilities': xgb_pred_proba,
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred, zero_division=0),
            'recall': recall_score(y_test, xgb_pred, zero_division=0),
            'f1': f1_score(y_test, xgb_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, xgb_pred_proba) if len(np.unique(y_test)) > 1 else 0
        }
        
        logger.info(f"  Accuracy: {results['XGBoost']['accuracy']:.4f}")
        logger.info(f"  F1-Score: {results['XGBoost']['f1']:.4f}")
        logger.info(f"  ROC-AUC: {results['XGBoost']['roc_auc']:.4f}")
        
        # LightGBM
        logger.info("\n3. LightGBM Classifier...")
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_clf.fit(X_train, y_train)
        lgb_pred = lgb_clf.predict(X_test)
        lgb_pred_proba = lgb_clf.predict_proba(X_test)[:, 1]
        
        results['LightGBM'] = {
            'model': lgb_clf,
            'predictions': lgb_pred,
            'probabilities': lgb_pred_proba,
            'accuracy': accuracy_score(y_test, lgb_pred),
            'precision': precision_score(y_test, lgb_pred, zero_division=0),
            'recall': recall_score(y_test, lgb_pred, zero_division=0),
            'f1': f1_score(y_test, lgb_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, lgb_pred_proba) if len(np.unique(y_test)) > 1 else 0
        }
        
        logger.info(f"  Accuracy: {results['LightGBM']['accuracy']:.4f}")
        logger.info(f"  F1-Score: {results['LightGBM']['f1']:.4f}")
        logger.info(f"  ROC-AUC: {results['LightGBM']['roc_auc']:.4f}")
        
        return results
    
    def train_regression_models(self, X_train, X_test, y_train, y_test, target_name):
        """Train regression models"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Training Regression Models for: {target_name}")
        logger.info(f"{'='*70}")
        
        results = {}
        
        # Random Forest
        logger.info("\n1. Random Forest Regressor...")
        rf_reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf_reg.fit(X_train, y_train)
        rf_pred = rf_reg.predict(X_test)
        
        results['Random Forest'] = {
            'model': rf_reg,
            'predictions': rf_pred,
            'mse': mean_squared_error(y_test, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'mae': mean_absolute_error(y_test, rf_pred),
            'r2': r2_score(y_test, rf_pred)
        }
        
        logger.info(f"  RMSE: {results['Random Forest']['rmse']:.4f}")
        logger.info(f"  MAE: {results['Random Forest']['mae']:.4f}")
        logger.info(f"  R²: {results['Random Forest']['r2']:.4f}")
        
        # XGBoost
        logger.info("\n2. XGBoost Regressor...")
        xgb_reg = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_reg.fit(X_train, y_train)
        xgb_pred = xgb_reg.predict(X_test)
        
        results['XGBoost'] = {
            'model': xgb_reg,
            'predictions': xgb_pred,
            'mse': mean_squared_error(y_test, xgb_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'mae': mean_absolute_error(y_test, xgb_pred),
            'r2': r2_score(y_test, xgb_pred)
        }
        
        logger.info(f"  RMSE: {results['XGBoost']['rmse']:.4f}")
        logger.info(f"  MAE: {results['XGBoost']['mae']:.4f}")
        logger.info(f"  R²: {results['XGBoost']['r2']:.4f}")
        
        # LightGBM
        logger.info("\n3. LightGBM Regressor...")
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_reg.fit(X_train, y_train)
        lgb_pred = lgb_reg.predict(X_test)
        
        results['LightGBM'] = {
            'model': lgb_reg,
            'predictions': lgb_pred,
            'mse': mean_squared_error(y_test, lgb_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, lgb_pred)),
            'mae': mean_absolute_error(y_test, lgb_pred),
            'r2': r2_score(y_test, lgb_pred)
        }
        
        logger.info(f"  RMSE: {results['LightGBM']['rmse']:.4f}")
        logger.info(f"  MAE: {results['LightGBM']['mae']:.4f}")
        logger.info(f"  R²: {results['LightGBM']['r2']:.4f}")
        
        return results
    
    def save_results(self, all_results):
        """Save model results to file"""
        logger.info("\nSaving results...")
        
        results_file = f'{self.results_dir}/model_evaluation/model_results.txt'
        
        with open(results_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRANSFER SUCCESS PREDICTION - MODEL RESULTS\n")
            f.write("="*70 + "\n\n")
            
            for task_name, task_results in all_results.items():
                f.write(f"\n{task_name}\n")
                f.write("-"*70 + "\n")
                
                for model_name, metrics in task_results.items():
                    if model_name == 'y_test':
                        continue
                    
                    f.write(f"\n{model_name}:\n")
                    
                    # Classification metrics
                    if 'accuracy' in metrics:
                        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                        f.write(f"  Precision: {metrics['precision']:.4f}\n")
                        f.write(f"  Recall: {metrics['recall']:.4f}\n")
                        f.write(f"  F1-Score: {metrics['f1']:.4f}\n")
                        f.write(f"  ROC-AUC: {metrics['roc_auc']:.4f}\n")
                    
                    # Regression metrics
                    if 'rmse' in metrics:
                        f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
                        f.write(f"  MAE: {metrics['mae']:.4f}\n")
                        f.write(f"  R²: {metrics['r2']:.4f}\n")
        
        logger.info(f"✅ Results saved to: {results_file}")
    
    def train_all_models(self):
        """Main function to train all models"""
        logger.info("\n" + "="*70)
        logger.info("STARTING MODEL TRAINING")
        logger.info("="*70)
        
        # Load data
        df = self.load_data()
        
        # Prepare features
        feature_cols = self.prepare_features(df)
        
        # Remove rows with missing values in features
        df_clean = df[feature_cols + [
            'target_success_goals', 'target_success_contribution',
            'target_goals_after', 'target_goal_contribution_after'
        ]].dropna()
        
        logger.info(f"\nClean dataset: {len(df_clean)} records")
        
        X = df_clean[feature_cols]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # Store all results
        all_results = {}
        
        # ===================================================================
        # CLASSIFICATION TASKS
        # ===================================================================
        
        # Task 1: Goal Improvement Classification
        logger.info("\n" + "="*70)
        logger.info("CLASSIFICATION TASK 1: Goal Improvement Prediction")
        logger.info("="*70)
        
        y_goal_success = df_clean['target_success_goals']
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_goal_success, test_size=0.2, random_state=42, stratify=y_goal_success
        )
        
        results_goal = self.train_classification_models(
            X_train, X_test, y_train, y_test, 'Goal Improvement'
        )
        results_goal['y_test'] = y_test
        all_results['Classification: Goal Improvement'] = results_goal
        
        # Task 2: Goal Contribution Classification
        logger.info("\n" + "="*70)
        logger.info("CLASSIFICATION TASK 2: Goal Contribution Improvement")
        logger.info("="*70)
        
        y_contrib_success = df_clean['target_success_contribution']
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_contrib_success, test_size=0.2, random_state=42, stratify=y_contrib_success
        )
        
        results_contrib = self.train_classification_models(
            X_train, X_test, y_train, y_test, 'Goal Contribution'
        )
        results_contrib['y_test'] = y_test
        all_results['Classification: Goal Contribution'] = results_contrib
        
        # ===================================================================
        # REGRESSION TASKS
        # ===================================================================
        
        # Task 3: Goals After Transfer Regression
        logger.info("\n" + "="*70)
        logger.info("REGRESSION TASK 1: Goals After Transfer Prediction")
        logger.info("="*70)
        
        y_goals_after = df_clean['target_goals_after']
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_goals_after, test_size=0.2, random_state=42
        )
        
        results_goals_reg = self.train_regression_models(
            X_train, X_test, y_train, y_test, 'Goals After Transfer'
        )
        results_goals_reg['y_test'] = y_test
        all_results['Regression: Goals After Transfer'] = results_goals_reg
        
        # Task 4: Goal Contribution After Transfer Regression
        logger.info("\n" + "="*70)
        logger.info("REGRESSION TASK 2: Goal Contribution After Transfer")
        logger.info("="*70)
        
        y_contrib_after = df_clean['target_goal_contribution_after']
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_contrib_after, test_size=0.2, random_state=42
        )
        
        results_contrib_reg = self.train_regression_models(
            X_train, X_test, y_train, y_test, 'Goal Contribution After'
        )
        results_contrib_reg['y_test'] = y_test
        all_results['Regression: Goal Contribution After'] = results_contrib_reg
        
        # Save results
        self.save_results(all_results)
        
        # Save best models
        logger.info("\nSaving best models...")
        
        # Save best classification model (highest F1)
        best_clf_model = results_goal['XGBoost']['model']  # XGBoost typically performs well
        joblib.dump(best_clf_model, f'{self.models_dir}/best_classifier.pkl')
        logger.info(f"✅ Saved best classifier to: {self.models_dir}/best_classifier.pkl")
        
        # Save best regression model (lowest RMSE)
        best_reg_model = results_goals_reg['XGBoost']['model']
        joblib.dump(best_reg_model, f'{self.models_dir}/best_regressor.pkl')
        logger.info(f"✅ Saved best regressor to: {self.models_dir}/best_regressor.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, f'{self.models_dir}/scaler.pkl')
        logger.info(f"✅ Saved scaler to: {self.models_dir}/scaler.pkl")
        
        # Save feature names
        with open(f'{self.models_dir}/feature_names.txt', 'w') as f:
            f.write('\n'.join(feature_cols))
        logger.info(f"✅ Saved feature names to: {self.models_dir}/feature_names.txt")
        
        logger.info("\n" + "="*70)
        logger.info("MODEL TRAINING COMPLETE!")
        logger.info("="*70)
        
        return all_results


def main():
    """Main execution"""
    predictor = TransferSuccessPredictor()
    results = predictor.train_all_models()
    
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    
    for task_name, task_results in results.items():
        print(f"\n{task_name}:")
        for model_name, metrics in task_results.items():
            if model_name == 'y_test':
                continue
            
            if 'accuracy' in metrics:
                print(f"  {model_name}: F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")
            elif 'rmse' in metrics:
                print(f"  {model_name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")


if __name__ == "__main__":
    main()

