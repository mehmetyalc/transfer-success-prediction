"""
Neural Network Models for Transfer Success Prediction
Using TensorFlow/Keras
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NeuralNetworkModels:
    """Create and train neural network models"""
    
    def __init__(self):
        self.data_path = 'data/processed/transfers_ml_ready.csv'
        self.models_dir = 'models/neural_networks'
        self.results_dir = 'results/neural_networks'
        self.scaler = StandardScaler()
        
        # Create directories
        import os
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set random seeds
        np.random.seed(42)
        tf.random.set_seed(42)
    
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
    
    def create_classifier_model(self, input_dim):
        """Create neural network classifier"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def create_regressor_model(self, input_dim):
        """Create neural network regressor"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        return model
    
    def train_neural_networks(self):
        """Train all neural network models"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING NEURAL NETWORK MODELS")
        logger.info("="*70)
        
        # Load data
        df_clean, feature_cols = self.load_and_prepare_data()
        
        X = df_clean[feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        # ===================================================================
        # CLASSIFICATION TASK
        # ===================================================================
        
        logger.info("\n" + "="*70)
        logger.info("NEURAL NETWORK CLASSIFIER")
        logger.info("="*70)
        
        y_clf = df_clean['target_success_goals'].values
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_scaled, y_clf, test_size=0.2, random_state=42, stratify=y_clf
        )
        
        # Create model
        nn_clf = self.create_classifier_model(X_scaled.shape[1])
        
        logger.info("\nModel Architecture:")
        nn_clf.summary(print_fn=logger.info)
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001
        )
        
        # Train
        logger.info("\nTraining classifier...")
        history_clf = nn_clf.fit(
            X_train_clf, y_train_clf,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        y_pred_proba_clf = nn_clf.predict(X_test_clf).flatten()
        y_pred_clf = (y_pred_proba_clf > 0.5).astype(int)
        
        results['NN_Classifier'] = {
            'model': nn_clf,
            'history': history_clf,
            'accuracy': accuracy_score(y_test_clf, y_pred_clf),
            'f1': f1_score(y_test_clf, y_pred_clf),
            'roc_auc': roc_auc_score(y_test_clf, y_pred_proba_clf)
        }
        
        logger.info(f"\nClassifier Results:")
        logger.info(f"  Accuracy: {results['NN_Classifier']['accuracy']:.4f}")
        logger.info(f"  F1-Score: {results['NN_Classifier']['f1']:.4f}")
        logger.info(f"  ROC-AUC: {results['NN_Classifier']['roc_auc']:.4f}")
        logger.info(f"  Training epochs: {len(history_clf.history['loss'])}")
        
        # ===================================================================
        # REGRESSION TASK
        # ===================================================================
        
        logger.info("\n" + "="*70)
        logger.info("NEURAL NETWORK REGRESSOR")
        logger.info("="*70)
        
        y_reg = df_clean['target_goals_after'].values
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_scaled, y_reg, test_size=0.2, random_state=42
        )
        
        # Create model
        nn_reg = self.create_regressor_model(X_scaled.shape[1])
        
        logger.info("\nModel Architecture:")
        nn_reg.summary(print_fn=logger.info)
        
        # Callbacks
        early_stopping_reg = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr_reg = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001
        )
        
        # Train
        logger.info("\nTraining regressor...")
        history_reg = nn_reg.fit(
            X_train_reg, y_train_reg,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping_reg, reduce_lr_reg],
            verbose=0
        )
        
        # Evaluate
        y_pred_reg = nn_reg.predict(X_test_reg).flatten()
        
        results['NN_Regressor'] = {
            'model': nn_reg,
            'history': history_reg,
            'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)),
            'r2': r2_score(y_test_reg, y_pred_reg)
        }
        
        logger.info(f"\nRegressor Results:")
        logger.info(f"  RMSE: {results['NN_Regressor']['rmse']:.4f}")
        logger.info(f"  R²: {results['NN_Regressor']['r2']:.4f}")
        logger.info(f"  Training epochs: {len(history_reg.history['loss'])}")
        
        # Save models
        self.save_models(results)
        self.save_results(results)
        
        logger.info("\n" + "="*70)
        logger.info("NEURAL NETWORK TRAINING COMPLETE!")
        logger.info("="*70)
        
        return results
    
    def save_models(self, results):
        """Save neural network models"""
        logger.info("\nSaving neural network models...")
        
        # Save classifier
        results['NN_Classifier']['model'].save(f"{self.models_dir}/nn_classifier.keras")
        logger.info(f"✅ Saved: {self.models_dir}/nn_classifier.keras")
        
        # Save regressor
        results['NN_Regressor']['model'].save(f"{self.models_dir}/nn_regressor.keras")
        logger.info(f"✅ Saved: {self.models_dir}/nn_regressor.keras")
        
        # Save scaler
        joblib.dump(self.scaler, f"{self.models_dir}/scaler_nn.pkl")
        logger.info(f"✅ Saved: {self.models_dir}/scaler_nn.pkl")
        
        # Save training history
        history_clf = pd.DataFrame(results['NN_Classifier']['history'].history)
        history_clf.to_csv(f"{self.results_dir}/classifier_training_history.csv", index=False)
        
        history_reg = pd.DataFrame(results['NN_Regressor']['history'].history)
        history_reg.to_csv(f"{self.results_dir}/regressor_training_history.csv", index=False)
        
        logger.info(f"✅ Saved training histories")
    
    def save_results(self, results):
        """Save neural network results"""
        logger.info("\nSaving results...")
        
        results_file = f'{self.results_dir}/nn_results.txt'
        
        with open(results_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("NEURAL NETWORK MODELS RESULTS\n")
            f.write("="*70 + "\n\n")
            
            f.write("NN_Classifier:\n")
            f.write("-"*70 + "\n")
            f.write(f"  Accuracy: {results['NN_Classifier']['accuracy']:.4f}\n")
            f.write(f"  F1-Score: {results['NN_Classifier']['f1']:.4f}\n")
            f.write(f"  ROC-AUC: {results['NN_Classifier']['roc_auc']:.4f}\n")
            f.write(f"  Training epochs: {len(results['NN_Classifier']['history'].history['loss'])}\n")
            
            f.write("\nNN_Regressor:\n")
            f.write("-"*70 + "\n")
            f.write(f"  RMSE: {results['NN_Regressor']['rmse']:.4f}\n")
            f.write(f"  R²: {results['NN_Regressor']['r2']:.4f}\n")
            f.write(f"  Training epochs: {len(results['NN_Regressor']['history'].history['loss'])}\n")
        
        logger.info(f"✅ Results saved to: {results_file}")


def main():
    """Main execution"""
    nn_models = NeuralNetworkModels()
    results = nn_models.train_neural_networks()
    
    print("\n" + "="*70)
    print("NEURAL NETWORK MODELS SUMMARY")
    print("="*70)
    
    print(f"\nNN_Classifier:")
    print(f"  F1: {results['NN_Classifier']['f1']:.4f}, AUC: {results['NN_Classifier']['roc_auc']:.4f}")
    
    print(f"\nNN_Regressor:")
    print(f"  RMSE: {results['NN_Regressor']['rmse']:.4f}, R²: {results['NN_Regressor']['r2']:.4f}")


if __name__ == "__main__":
    main()

