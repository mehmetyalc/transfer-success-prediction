"""
Ensemble Methods for Transfer Success Prediction
Voting, Stacking, and Model Combination
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    RandomForestClassifier, RandomForestRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnsembleModels:
    """Create and train ensemble models"""
    
    def __init__(self):
        self.data_path = 'data/processed/transfers_ml_ready.csv'
        self.models_dir = 'models/ensemble'
        self.results_dir = 'results/ensemble'
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
    
    def create_voting_classifier(self):
        """Create Voting Classifier"""
        logger.info("\nCreating Voting Classifier...")
        
        # Base models
        rf_clf = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
        
        # Voting classifier (soft voting for probability-based)
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', rf_clf),
                ('xgb', xgb_clf),
                ('lgb', lgb_clf)
            ],
            voting='soft'
        )
        
        return voting_clf
    
    def create_stacking_classifier(self):
        """Create Stacking Classifier"""
        logger.info("\nCreating Stacking Classifier...")
        
        # Base models
        rf_clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        )
        
        # Meta-learner
        meta_clf = LogisticRegression(random_state=42, max_iter=1000)
        
        # Stacking classifier
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', rf_clf),
                ('xgb', xgb_clf),
                ('lgb', lgb_clf)
            ],
            final_estimator=meta_clf,
            cv=5
        )
        
        return stacking_clf
    
    def create_voting_regressor(self):
        """Create Voting Regressor"""
        logger.info("\nCreating Voting Regressor...")
        
        # Base models
        rf_reg = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        
        xgb_reg = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
        
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
        
        # Voting regressor
        voting_reg = VotingRegressor(
            estimators=[
                ('rf', rf_reg),
                ('xgb', xgb_reg),
                ('lgb', lgb_reg)
            ]
        )
        
        return voting_reg
    
    def create_stacking_regressor(self):
        """Create Stacking Regressor"""
        logger.info("\nCreating Stacking Regressor...")
        
        # Base models
        rf_reg = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        
        xgb_reg = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1
        )
        
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        )
        
        # Meta-learner
        meta_reg = Ridge(random_state=42)
        
        # Stacking regressor
        stacking_reg = StackingRegressor(
            estimators=[
                ('rf', rf_reg),
                ('xgb', xgb_reg),
                ('lgb', lgb_reg)
            ],
            final_estimator=meta_reg,
            cv=5
        )
        
        return stacking_reg
    
    def train_ensemble_models(self):
        """Train all ensemble models"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING ENSEMBLE MODELS")
        logger.info("="*70)
        
        # Load data
        df_clean, feature_cols = self.load_and_prepare_data()
        
        X = df_clean[feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        results = {}
        
        # ===================================================================
        # CLASSIFICATION ENSEMBLES
        # ===================================================================
        
        logger.info("\n" + "="*70)
        logger.info("CLASSIFICATION ENSEMBLES")
        logger.info("="*70)
        
        y_clf = df_clean['target_success_goals']
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_scaled, y_clf, test_size=0.2, random_state=42, stratify=y_clf
        )
        
        # Voting Classifier
        logger.info("\n1. Training Voting Classifier...")
        voting_clf = self.create_voting_classifier()
        voting_clf.fit(X_train_clf, y_train_clf)
        
        y_pred_voting = voting_clf.predict(X_test_clf)
        y_pred_proba_voting = voting_clf.predict_proba(X_test_clf)[:, 1]
        
        results['Voting_Classifier'] = {
            'model': voting_clf,
            'accuracy': accuracy_score(y_test_clf, y_pred_voting),
            'f1': f1_score(y_test_clf, y_pred_voting),
            'roc_auc': roc_auc_score(y_test_clf, y_pred_proba_voting)
        }
        
        logger.info(f"  Accuracy: {results['Voting_Classifier']['accuracy']:.4f}")
        logger.info(f"  F1-Score: {results['Voting_Classifier']['f1']:.4f}")
        logger.info(f"  ROC-AUC: {results['Voting_Classifier']['roc_auc']:.4f}")
        
        # Stacking Classifier
        logger.info("\n2. Training Stacking Classifier...")
        stacking_clf = self.create_stacking_classifier()
        stacking_clf.fit(X_train_clf, y_train_clf)
        
        y_pred_stacking = stacking_clf.predict(X_test_clf)
        y_pred_proba_stacking = stacking_clf.predict_proba(X_test_clf)[:, 1]
        
        results['Stacking_Classifier'] = {
            'model': stacking_clf,
            'accuracy': accuracy_score(y_test_clf, y_pred_stacking),
            'f1': f1_score(y_test_clf, y_pred_stacking),
            'roc_auc': roc_auc_score(y_test_clf, y_pred_proba_stacking)
        }
        
        logger.info(f"  Accuracy: {results['Stacking_Classifier']['accuracy']:.4f}")
        logger.info(f"  F1-Score: {results['Stacking_Classifier']['f1']:.4f}")
        logger.info(f"  ROC-AUC: {results['Stacking_Classifier']['roc_auc']:.4f}")
        
        # ===================================================================
        # REGRESSION ENSEMBLES
        # ===================================================================
        
        logger.info("\n" + "="*70)
        logger.info("REGRESSION ENSEMBLES")
        logger.info("="*70)
        
        y_reg = df_clean['target_goals_after']
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_scaled, y_reg, test_size=0.2, random_state=42
        )
        
        # Voting Regressor
        logger.info("\n1. Training Voting Regressor...")
        voting_reg = self.create_voting_regressor()
        voting_reg.fit(X_train_reg, y_train_reg)
        
        y_pred_voting_reg = voting_reg.predict(X_test_reg)
        
        results['Voting_Regressor'] = {
            'model': voting_reg,
            'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_voting_reg)),
            'r2': r2_score(y_test_reg, y_pred_voting_reg)
        }
        
        logger.info(f"  RMSE: {results['Voting_Regressor']['rmse']:.4f}")
        logger.info(f"  R²: {results['Voting_Regressor']['r2']:.4f}")
        
        # Stacking Regressor
        logger.info("\n2. Training Stacking Regressor...")
        stacking_reg = self.create_stacking_regressor()
        stacking_reg.fit(X_train_reg, y_train_reg)
        
        y_pred_stacking_reg = stacking_reg.predict(X_test_reg)
        
        results['Stacking_Regressor'] = {
            'model': stacking_reg,
            'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_stacking_reg)),
            'r2': r2_score(y_test_reg, y_pred_stacking_reg)
        }
        
        logger.info(f"  RMSE: {results['Stacking_Regressor']['rmse']:.4f}")
        logger.info(f"  R²: {results['Stacking_Regressor']['r2']:.4f}")
        
        # Save models
        self.save_models(results)
        self.save_results(results)
        
        logger.info("\n" + "="*70)
        logger.info("ENSEMBLE TRAINING COMPLETE!")
        logger.info("="*70)
        
        return results
    
    def save_models(self, results):
        """Save ensemble models"""
        logger.info("\nSaving ensemble models...")
        
        for model_name, model_data in results.items():
            filename = f"{self.models_dir}/{model_name.lower()}.pkl"
            joblib.dump(model_data['model'], filename)
            logger.info(f"✅ Saved: {filename}")
        
        joblib.dump(self.scaler, f"{self.models_dir}/scaler_ensemble.pkl")
        logger.info(f"✅ Saved: {self.models_dir}/scaler_ensemble.pkl")
    
    def save_results(self, results):
        """Save ensemble results"""
        logger.info("\nSaving results...")
        
        results_file = f'{self.results_dir}/ensemble_results.txt'
        
        with open(results_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ENSEMBLE MODELS RESULTS\n")
            f.write("="*70 + "\n\n")
            
            for model_name, metrics in results.items():
                f.write(f"\n{model_name}:\n")
                f.write("-"*70 + "\n")
                
                if 'accuracy' in metrics:
                    f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1']:.4f}\n")
                    f.write(f"  ROC-AUC: {metrics['roc_auc']:.4f}\n")
                
                if 'rmse' in metrics:
                    f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
                    f.write(f"  R²: {metrics['r2']:.4f}\n")
        
        logger.info(f"✅ Results saved to: {results_file}")


def main():
    """Main execution"""
    ensemble = EnsembleModels()
    results = ensemble.train_ensemble_models()
    
    print("\n" + "="*70)
    print("ENSEMBLE MODELS SUMMARY")
    print("="*70)
    
    for model_name, metrics in results.items():
        if 'accuracy' in metrics:
            print(f"\n{model_name}:")
            print(f"  F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        elif 'rmse' in metrics:
            print(f"\n{model_name}:")
            print(f"  RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()

