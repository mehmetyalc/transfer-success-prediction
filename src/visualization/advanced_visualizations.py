"""
Advanced Visualizations for Transfer Success Prediction
ROC Curves, Precision-Recall, Learning Curves, Confusion Matrices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data():
    """Load and prepare data"""
    print("Loading data...")
    df = pd.read_csv('data/processed/transfers_ml_ready.csv')
    
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
    df_clean = df[feature_cols + ['target_success_goals', 'target_goals_after']].dropna()
    
    return df_clean, feature_cols


def plot_roc_curves():
    """Plot ROC curves for all classification models"""
    print("\n" + "="*70)
    print("CREATING ROC CURVES")
    print("="*70)
    
    df_clean, feature_cols = load_data()
    
    X = df_clean[feature_cols]
    y = df_clean['target_success_goals']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    models = {
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Transfer Success Classification', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = 'results/figures/10_roc_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_precision_recall_curves():
    """Plot Precision-Recall curves"""
    print("\n" + "="*70)
    print("CREATING PRECISION-RECALL CURVES")
    print("="*70)
    
    df_clean, feature_cols = load_data()
    
    X = df_clean[feature_cols]
    y = df_clean['target_success_goals']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    models = {
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        ax.plot(recall, precision, linewidth=2, label=f'{name} (AP = {avg_precision:.3f})')
    
    # Plot baseline
    baseline = y_test.sum() / len(y_test)
    ax.axhline(baseline, color='k', linestyle='--', linewidth=1, label=f'Baseline (AP = {baseline:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Transfer Success Classification', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = 'results/figures/11_precision_recall_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_confusion_matrices():
    """Plot confusion matrices for classification models"""
    print("\n" + "="*70)
    print("CREATING CONFUSION MATRICES")
    print("="*70)
    
    df_clean, feature_cols = load_data()
    
    X = df_clean[feature_cols]
    y = df_clean['target_success_goals']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    models = {
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Confusion Matrices - Transfer Success Classification', 
                 fontsize=16, fontweight='bold')
    
    for idx, (name, model) in enumerate(models.items()):
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Improvement', 'Improvement'])
        disp.plot(ax=axes[idx], cmap='Blues', values_format='d')
        axes[idx].set_title(name, fontsize=13, fontweight='bold')
        axes[idx].grid(False)
    
    plt.tight_layout()
    output_file = 'results/figures/12_confusion_matrices.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_learning_curves():
    """Plot learning curves to show model performance vs training size"""
    print("\n" + "="*70)
    print("CREATING LEARNING CURVES")
    print("="*70)
    
    df_clean, feature_cols = load_data()
    
    X = df_clean[feature_cols]
    y = df_clean['target_success_goals']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # XGBoost model
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss')
    
    print("Computing learning curve...")
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_scaled, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(train_sizes, train_mean, 'o-', color='#2E86AB', linewidth=2, label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='#2E86AB')
    
    ax.plot(train_sizes, val_mean, 'o-', color='#A23B72', linewidth=2, label='Cross-validation score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.2, color='#A23B72')
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Learning Curve - XGBoost Classifier', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = 'results/figures/13_learning_curve.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_residuals_analysis():
    """Plot residuals analysis for regression model"""
    print("\n" + "="*70)
    print("CREATING RESIDUALS ANALYSIS")
    print("="*70)
    
    df_clean, feature_cols = load_data()
    
    X = df_clean[feature_cols]
    y = df_clean['target_goals_after']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Residuals Analysis - Goals After Transfer Prediction', 
                 fontsize=16, fontweight='bold')
    
    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=30)
    axes[0, 0].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Goals', fontsize=11)
    axes[0, 0].set_ylabel('Residuals', fontsize=11)
    axes[0, 0].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals Distribution
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Actual vs Predicted
    axes[1, 1].scatter(y_test, y_pred, alpha=0.5, s=30)
    max_val = max(y_test.max(), y_pred.max())
    axes[1, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
    axes[1, 1].set_xlabel('Actual Goals', fontsize=11)
    axes[1, 1].set_ylabel('Predicted Goals', fontsize=11)
    axes[1, 1].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = 'results/figures/14_residuals_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def main():
    """Main execution"""
    print("="*70)
    print("CREATING ADVANCED VISUALIZATIONS")
    print("="*70)
    
    # Create all visualizations
    plot_roc_curves()
    plot_precision_recall_curves()
    plot_confusion_matrices()
    plot_learning_curves()
    plot_residuals_analysis()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  - 10_roc_curves.png")
    print("  - 11_precision_recall_curves.png")
    print("  - 12_confusion_matrices.png")
    print("  - 13_learning_curve.png")
    print("  - 14_residuals_analysis.png")


if __name__ == "__main__":
    main()

