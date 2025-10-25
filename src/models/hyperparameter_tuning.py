"""
Hyperparameter Tuning for Transfer Success Prediction Models
Using GridSearchCV to find optimal parameters
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error, r2_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
import warnings
import json
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Tune hyperparameters for all models"""
    
    def __init__(self):
        self.data_path = 'data/processed/transfers_ml_ready.csv'
        self.models_dir = 'models/tuned'
        self.results_dir = 'results/hyperparameter_tuning'
        self.scaler = StandardScaler()
        
        # Create directories
        import os
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_and_prepare_data(self):
        """Load and prepare data"""
        logger.info("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Define feature columns
        exclude_cols = [
            'club_name', 'player_name', 'club_involved_name', 'fee', 'transfer_movement',
            'transfer_period', 'league_name', 'year', 'season', 'country', 'position',
            'position_group', 'age_category', 'fee_category',
            'target_goals_after', 'target_assists_after', 'target_goal_contribution_after',
            'target_minutes_after', 'target_success_goals', 'target_success_minutes',
            'target_success_composite', 'target_success_contribution',
            'perf_after_goals', 'perf_after_assists', 'perf_after_minutes', 'perf_after_matches',
            'goal_change', 'assist_change', 'minutes_change',
            'goal_contribution_after', 'goals_per_90_after', 'assists_per_90_after',
            'minutes_per_match_after', 'success_goals', 'success_minutes', 'success_composite'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Clean data
        df_clean = df[feature_cols + [
            'target_success_goals', 'target_goals_after'
        ]].dropna()
        
        logger.info(f"Clean dataset: {len(df_clean)} records, {len(feature_cols)} features")
        
        return df_clean, feature_cols
    
    def tune_random_forest_classifier(self, X_train, y_train):
        """Tune Random Forest Classifier"""
        logger.info("\n" + "="*70)
        logger.info("TUNING RANDOM FOREST CLASSIFIER")
        logger.info("="*70)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 5],
            'max_features': ['sqrt']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        
        logger.info("Starting grid search...")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"\nBest parameters: {grid_search.best_params_}")
        logger.info(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def tune_xgboost_classifier(self, X_train, y_train):
        """Tune XGBoost Classifier"""
        logger.info("\n" + "="*70)
        logger.info("TUNING XGBOOST CLASSIFIER")
        logger.info("="*70)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'min_child_weight': [1, 3]
        }
        
        xgb_clf = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
        
        grid_search = GridSearchCV(
            xgb_clf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        
        logger.info("Starting grid search...")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"\nBest parameters: {grid_search.best_params_}")
        logger.info(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def tune_lightgbm_classifier(self, X_train, y_train):
        """Tune LightGBM Classifier"""
        logger.info("\n" + "="*70)
        logger.info("TUNING LIGHTGBM CLASSIFIER")
        logger.info("="*70)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [20, 31],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        
        lgb_clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        
        grid_search = GridSearchCV(
            lgb_clf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        
        logger.info("Starting grid search...")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"\nBest parameters: {grid_search.best_params_}")
        logger.info(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def tune_xgboost_regressor(self, X_train, y_train):
        """Tune XGBoost Regressor"""
        logger.info("\n" + "="*70)
        logger.info("TUNING XGBOOST REGRESSOR")
        logger.info("="*70)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'min_child_weight': [1, 3]
        }
        
        xgb_reg = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        # Use negative MSE as scoring
        grid_search = GridSearchCV(
            xgb_reg, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
        )
        
        logger.info("Starting grid search...")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"\nBest parameters: {grid_search.best_params_}")
        logger.info(f"Best MSE: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, -grid_search.best_score_
    
    def run_tuning(self):
        """Main function to run all tuning"""
        logger.info("\n" + "="*70)
        logger.info("STARTING HYPERPARAMETER TUNING")
        logger.info("="*70)
        
        # Load data
        df_clean, feature_cols = self.load_and_prepare_data()
        
        X = df_clean[feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        results = {}
        
        # ===================================================================
        # CLASSIFICATION TASK: Goal Improvement
        # ===================================================================
        
        logger.info("\n" + "="*70)
        logger.info("CLASSIFICATION TASK: Goal Improvement")
        logger.info("="*70)
        
        y_clf = df_clean['target_success_goals']
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_scaled, y_clf, test_size=0.2, random_state=42, stratify=y_clf
        )
        
        # Tune Random Forest
        rf_clf, rf_params, rf_score = self.tune_random_forest_classifier(X_train_clf, y_train_clf)
        results['RF_Classifier'] = {
            'model': rf_clf,
            'params': rf_params,
            'cv_score': rf_score,
            'test_f1': f1_score(y_test_clf, rf_clf.predict(X_test_clf))
        }
        
        # Tune XGBoost
        xgb_clf, xgb_params, xgb_score = self.tune_xgboost_classifier(X_train_clf, y_train_clf)
        results['XGB_Classifier'] = {
            'model': xgb_clf,
            'params': xgb_params,
            'cv_score': xgb_score,
            'test_f1': f1_score(y_test_clf, xgb_clf.predict(X_test_clf))
        }
        
        # Tune LightGBM
        lgb_clf, lgb_params, lgb_score = self.tune_lightgbm_classifier(X_train_clf, y_train_clf)
        results['LGB_Classifier'] = {
            'model': lgb_clf,
            'params': lgb_params,
            'cv_score': lgb_score,
            'test_f1': f1_score(y_test_clf, lgb_clf.predict(X_test_clf))
        }
        
        # ===================================================================
        # REGRESSION TASK: Goals After Transfer
        # ===================================================================
        
        logger.info("\n" + "="*70)
        logger.info("REGRESSION TASK: Goals After Transfer")
        logger.info("="*70)
        
        y_reg = df_clean['target_goals_after']
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_scaled, y_reg, test_size=0.2, random_state=42
        )
        
        # Tune XGBoost Regressor
        xgb_reg, xgb_reg_params, xgb_reg_score = self.tune_xgboost_regressor(X_train_reg, y_train_reg)
        y_pred_reg = xgb_reg.predict(X_test_reg)
        results['XGB_Regressor'] = {
            'model': xgb_reg,
            'params': xgb_reg_params,
            'cv_mse': xgb_reg_score,
            'test_rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)),
            'test_r2': r2_score(y_test_reg, y_pred_reg)
        }
        
        # Save results
        self.save_results(results)
        
        # Save best models
        logger.info("\n" + "="*70)
        logger.info("SAVING TUNED MODELS")
        logger.info("="*70)
        
        joblib.dump(results['XGB_Classifier']['model'], f'{self.models_dir}/xgb_classifier_tuned.pkl')
        joblib.dump(results['XGB_Regressor']['model'], f'{self.models_dir}/xgb_regressor_tuned.pkl')
        joblib.dump(self.scaler, f'{self.models_dir}/scaler_tuned.pkl')
        
        logger.info("✅ Saved tuned models")
        
        logger.info("\n" + "="*70)
        logger.info("HYPERPARAMETER TUNING COMPLETE!")
        logger.info("="*70)
        
        return results
    
    def save_results(self, results):
        """Save tuning results"""
        logger.info("\nSaving tuning results...")
        
        results_file = f'{self.results_dir}/tuning_results.txt'
        
        with open(results_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("HYPERPARAMETER TUNING RESULTS\n")
            f.write("="*70 + "\n\n")
            
            for model_name, model_results in results.items():
                if model_name.endswith('Classifier'):
                    f.write(f"\n{model_name}:\n")
                    f.write("-"*70 + "\n")
                    f.write(f"Best Parameters:\n")
                    for param, value in model_results['params'].items():
                        f.write(f"  {param}: {value}\n")
                    f.write(f"\nCross-Validation F1: {model_results['cv_score']:.4f}\n")
                    f.write(f"Test F1: {model_results['test_f1']:.4f}\n")
                
                elif model_name.endswith('Regressor'):
                    f.write(f"\n{model_name}:\n")
                    f.write("-"*70 + "\n")
                    f.write(f"Best Parameters:\n")
                    for param, value in model_results['params'].items():
                        f.write(f"  {param}: {value}\n")
                    f.write(f"\nCross-Validation MSE: {model_results['cv_mse']:.4f}\n")
                    f.write(f"Test RMSE: {model_results['test_rmse']:.4f}\n")
                    f.write(f"Test R²: {model_results['test_r2']:.4f}\n")
        
        logger.info(f"✅ Results saved to: {results_file}")
        
        # Save parameters as JSON
        params_dict = {k: v['params'] for k, v in results.items()}
        with open(f'{self.results_dir}/best_params.json', 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        logger.info(f"✅ Parameters saved to: {self.results_dir}/best_params.json")


def main():
    """Main execution"""
    tuner = HyperparameterTuner()
    results = tuner.run_tuning()
    
    print("\n" + "="*70)
    print("TUNING SUMMARY")
    print("="*70)
    
    for model_name, model_results in results.items():
        if model_name.endswith('Classifier'):
            print(f"\n{model_name}:")
            print(f"  CV F1: {model_results['cv_score']:.4f}")
            print(f"  Test F1: {model_results['test_f1']:.4f}")
        elif model_name.endswith('Regressor'):
            print(f"\n{model_name}:")
            print(f"  CV MSE: {model_results['cv_mse']:.4f}")
            print(f"  Test RMSE: {model_results['test_rmse']:.4f}")
            print(f"  Test R²: {model_results['test_r2']:.4f}")


if __name__ == "__main__":
    main()

