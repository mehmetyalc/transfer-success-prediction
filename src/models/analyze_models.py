"""
Analyze trained models and create feature importance visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_models_and_data():
    """Load trained models and data"""
    print("Loading models and data...")
    
    # Load models
    clf_model = joblib.load('models/best_classifier.pkl')
    reg_model = joblib.load('models/best_regressor.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Load feature names
    with open('models/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Load data
    df = pd.read_csv('data/processed/transfers_ml_ready.csv')
    
    return clf_model, reg_model, scaler, feature_names, df


def plot_feature_importance(model, feature_names, title, output_file):
    """Plot feature importance"""
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print(f"Model doesn't have feature_importances_ attribute")
        return
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    top_n = 20
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(top_features)), top_features['importance'].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'{title}\nTop {top_n} Most Important Features', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Color bars by importance
    colors = plt.cm.viridis(top_features['importance'].values / top_features['importance'].values.max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()
    
    return importance_df


def create_model_comparison_plot():
    """Create model comparison visualization"""
    
    # Model results from training
    results = {
        'Goal Improvement\n(Classification)': {
            'Random Forest': {'F1': 0.6957, 'AUC': 0.9342},
            'XGBoost': {'F1': 0.7925, 'AUC': 0.9564},
            'LightGBM': {'F1': 0.8077, 'AUC': 0.9570}
        },
        'Goal Contribution\n(Classification)': {
            'Random Forest': {'F1': 0.9630, 'AUC': 1.0000},
            'XGBoost': {'F1': 1.0000, 'AUC': 1.0000},
            'LightGBM': {'F1': 1.0000, 'AUC': 1.0000}
        },
        'Goals After\n(Regression)': {
            'Random Forest': {'R²': 0.7791, 'RMSE': 1.3731},
            'XGBoost': {'R²': 0.8153, 'RMSE': 1.2556},
            'LightGBM': {'R²': 0.7652, 'RMSE': 1.4155}
        },
        'Goal Contribution After\n(Regression)': {
            'Random Forest': {'R²': 0.9233, 'RMSE': 1.2188},
            'XGBoost': {'R²': 0.9293, 'RMSE': 1.1707},
            'LightGBM': {'R²': 0.8453, 'RMSE': 1.7311}
        }
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    tasks = list(results.keys())
    
    for idx, (task, ax) in enumerate(zip(tasks, axes.flatten())):
        task_results = results[task]
        models = list(task_results.keys())
        
        if 'Classification' in task:
            # Plot F1 and AUC
            f1_scores = [task_results[m]['F1'] for m in models]
            auc_scores = [task_results[m]['AUC'] for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, f1_scores, width, label='F1-Score', alpha=0.8)
            bars2 = ax.bar(x + width/2, auc_scores, width, label='ROC-AUC', alpha=0.8)
            
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title(task, fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15, ha='right')
            ax.legend()
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        else:
            # Plot R² and RMSE
            r2_scores = [task_results[m]['R²'] for m in models]
            rmse_scores = [task_results[m]['RMSE'] for m in models]
            
            # Normalize RMSE to 0-1 scale for visualization
            rmse_normalized = [1 - (r / max(rmse_scores)) for r in rmse_scores]
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, r2_scores, width, label='R² Score', alpha=0.8)
            bars2 = ax.bar(x + width/2, rmse_normalized, width, label='RMSE (normalized)', alpha=0.8)
            
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title(task, fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15, ha='right')
            ax.legend()
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, r2 in zip(bars1, r2_scores):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{r2:.3f}', ha='center', va='bottom', fontsize=9)
            
            for bar, rmse in zip(bars2, rmse_scores):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{rmse:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = 'results/figures/06_model_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_prediction_analysis(clf_model, reg_model, scaler, feature_names, df):
    """Analyze predictions vs actual values"""
    
    # Prepare data
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
    
    df_clean = df[feature_names + ['target_goals_after', 'player_name']].dropna()
    
    X = df_clean[feature_names]
    y_actual = df_clean['target_goals_after']
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = reg_model.predict(X_scaled)
    
    # Create scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Prediction Analysis: Goals After Transfer', fontsize=16, fontweight='bold')
    
    # Scatter plot: Actual vs Predicted
    axes[0].scatter(y_actual, y_pred, alpha=0.5, s=30)
    max_val = max(y_actual.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual Goals', fontsize=12)
    axes[0].set_ylabel('Predicted Goals', fontsize=12)
    axes[0].set_title('Actual vs Predicted Goals', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_actual - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=30)
    axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Goals', fontsize=12)
    axes[1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = 'results/figures/07_prediction_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()
    
    # Find best and worst predictions
    df_clean['predicted_goals'] = y_pred
    df_clean['prediction_error'] = np.abs(residuals)
    
    print("\n" + "="*70)
    print("BEST PREDICTIONS (Lowest Error)")
    print("="*70)
    best = df_clean.nsmallest(10, 'prediction_error')[['player_name', 'target_goals_after', 'predicted_goals', 'prediction_error']]
    print(best.to_string(index=False))
    
    print("\n" + "="*70)
    print("WORST PREDICTIONS (Highest Error)")
    print("="*70)
    worst = df_clean.nlargest(10, 'prediction_error')[['player_name', 'target_goals_after', 'predicted_goals', 'prediction_error']]
    print(worst.to_string(index=False))


def main():
    """Main execution"""
    print("="*70)
    print("MODEL ANALYSIS AND VISUALIZATION")
    print("="*70)
    
    # Load models and data
    clf_model, reg_model, scaler, feature_names, df = load_models_and_data()
    
    print(f"\nLoaded:")
    print(f"  - Classifier: {type(clf_model).__name__}")
    print(f"  - Regressor: {type(reg_model).__name__}")
    print(f"  - Features: {len(feature_names)}")
    print(f"  - Records: {len(df)}")
    
    # Feature importance for classifier
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    clf_importance = plot_feature_importance(
        clf_model, feature_names,
        'Classification Model (Goal Improvement)',
        'results/figures/08_feature_importance_classifier.png'
    )
    
    # Feature importance for regressor
    reg_importance = plot_feature_importance(
        reg_model, feature_names,
        'Regression Model (Goals After Transfer)',
        'results/figures/09_feature_importance_regressor.png'
    )
    
    # Model comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    create_model_comparison_plot()
    
    # Prediction analysis
    print("\n" + "="*70)
    print("PREDICTION ANALYSIS")
    print("="*70)
    create_prediction_analysis(clf_model, reg_model, scaler, feature_names, df)
    
    # Save top features to file
    print("\n" + "="*70)
    print("SAVING TOP FEATURES")
    print("="*70)
    
    with open('results/model_evaluation/top_features.txt', 'w') as f:
        f.write("TOP 20 MOST IMPORTANT FEATURES\n")
        f.write("="*70 + "\n\n")
        
        f.write("Classification Model (Goal Improvement):\n")
        f.write("-"*70 + "\n")
        for idx, row in clf_importance.head(20).iterrows():
            f.write(f"{row['feature']:40s} {row['importance']:.6f}\n")
        
        f.write("\n\nRegression Model (Goals After Transfer):\n")
        f.write("-"*70 + "\n")
        for idx, row in reg_importance.head(20).iterrows():
            f.write(f"{row['feature']:40s} {row['importance']:.6f}\n")
    
    print("✅ Saved: results/model_evaluation/top_features.txt")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  - 06_model_comparison.png")
    print("  - 07_prediction_analysis.png")
    print("  - 08_feature_importance_classifier.png")
    print("  - 09_feature_importance_regressor.png")


if __name__ == "__main__":
    main()

