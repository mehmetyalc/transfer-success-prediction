"""
Compare model performance between baseline (821 records) and expanded (1,483 records) datasets
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Define baseline results (from 821-record training)
baseline_results = {
    "dataset_size": 821,
    "train_size": 656,  # 80% of 821
    "test_size": 165,   # 20% of 821
    "models": {
        "random_forest_clf": {
            "f1_score": 0.6957,
            "roc_auc": 0.9342
        },
        "xgboost_clf": {
            "f1_score": 0.7925,
            "roc_auc": 0.9564
        },
        "lightgbm_clf": {
            "f1_score": 0.8077,
            "roc_auc": 0.9570
        },
        "random_forest_reg": {
            "rmse": 1.3731,
            "r2": 0.7791
        },
        "xgboost_reg": {
            "rmse": 1.2556,
            "r2": 0.8153
        },
        "lightgbm_reg": {
            "rmse": 1.4155,
            "r2": 0.7652
        },
        "voting_classifier": {
            "f1_score": 0.7925,
            "roc_auc": 0.9577
        },
        "stacking_classifier": {
            "f1_score": 0.7692,
            "roc_auc": 0.9564
        },
        "voting_regressor": {
            "rmse": 1.2196,
            "r2": 0.8257
        },
        "stacking_regressor": {
            "rmse": 1.2224,
            "r2": 0.8249
        }
    }
}

# Load expanded results
with open('results/expanded_model_results.json', 'r') as f:
    expanded_results = json.load(f)

# Create comparison dataframe
comparison_data = []

for model_name in baseline_results['models'].keys():
    if model_name in expanded_results['models']:
        baseline_metrics = baseline_results['models'][model_name]
        expanded_metrics = expanded_results['models'][model_name]
        
        row = {
            'Model': model_name.replace('_', ' ').title(),
            'Type': 'Classification' if 'clf' in model_name else 'Regression'
        }
        
        # Add metrics based on model type
        if 'clf' in model_name:
            row['Baseline_F1'] = baseline_metrics['f1_score']
            row['Expanded_F1'] = expanded_metrics['f1_score']
            row['F1_Change'] = expanded_metrics['f1_score'] - baseline_metrics['f1_score']
            row['F1_Change_Pct'] = (row['F1_Change'] / baseline_metrics['f1_score']) * 100
            
            row['Baseline_AUC'] = baseline_metrics['roc_auc']
            row['Expanded_AUC'] = expanded_metrics['roc_auc']
            row['AUC_Change'] = expanded_metrics['roc_auc'] - baseline_metrics['roc_auc']
            row['AUC_Change_Pct'] = (row['AUC_Change'] / baseline_metrics['roc_auc']) * 100
        else:
            row['Baseline_RMSE'] = baseline_metrics['rmse']
            row['Expanded_RMSE'] = expanded_metrics['rmse']
            row['RMSE_Change'] = expanded_metrics['rmse'] - baseline_metrics['rmse']
            row['RMSE_Change_Pct'] = (row['RMSE_Change'] / baseline_metrics['rmse']) * 100
            
            row['Baseline_R2'] = baseline_metrics['r2']
            row['Expanded_R2'] = expanded_metrics['r2']
            row['R2_Change'] = expanded_metrics['r2'] - baseline_metrics['r2']
            row['R2_Change_Pct'] = (row['R2_Change'] / baseline_metrics['r2']) * 100 if baseline_metrics['r2'] != 0 else 0
        
        comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)

# Save comparison data
df_comparison.to_csv('results/model_performance_comparison.csv', index=False)
print("\n✅ Saved comparison data to: results/model_performance_comparison.csv")

# Print summary statistics
print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON: 821 vs 1,483 Records")
print("="*80)

print(f"\nDataset Size:")
print(f"  Baseline: {baseline_results['dataset_size']} records")
print(f"  Expanded: {expanded_results['dataset_size']} records")
print(f"  Increase: +{expanded_results['dataset_size'] - baseline_results['dataset_size']} records (+{((expanded_results['dataset_size'] - baseline_results['dataset_size']) / baseline_results['dataset_size'] * 100):.1f}%)")

print(f"\nFeatures:")
print(f"  Baseline: 69 features")
print(f"  Expanded: {expanded_results['n_features']} features")

# Classification models summary
clf_df = df_comparison[df_comparison['Type'] == 'Classification'].copy()
if not clf_df.empty:
    print("\n" + "-"*80)
    print("CLASSIFICATION MODELS (Goal Improvement Prediction)")
    print("-"*80)
    for _, row in clf_df.iterrows():
        print(f"\n{row['Model']}:")
        print(f"  F1-Score:  {row['Baseline_F1']:.4f} → {row['Expanded_F1']:.4f} ({row['F1_Change']:+.4f}, {row['F1_Change_Pct']:+.1f}%)")
        print(f"  ROC-AUC:   {row['Baseline_AUC']:.4f} → {row['Expanded_AUC']:.4f} ({row['AUC_Change']:+.4f}, {row['AUC_Change_Pct']:+.1f}%)")
        
        # Interpretation
        if row['F1_Change'] < -0.05:
            print(f"  ⚠️  Significant F1 degradation")
        elif row['F1_Change'] > 0.05:
            print(f"  ✅ Significant F1 improvement")
        else:
            print(f"  ➖ Minimal F1 change")

# Regression models summary
reg_df = df_comparison[df_comparison['Type'] == 'Regression'].copy()
if not reg_df.empty:
    print("\n" + "-"*80)
    print("REGRESSION MODELS (Goals After Transfer Prediction)")
    print("-"*80)
    for _, row in reg_df.iterrows():
        print(f"\n{row['Model']}:")
        print(f"  RMSE:  {row['Baseline_RMSE']:.4f} → {row['Expanded_RMSE']:.4f} ({row['RMSE_Change']:+.4f}, {row['RMSE_Change_Pct']:+.1f}%)")
        print(f"  R²:    {row['Baseline_R2']:.4f} → {row['Expanded_R2']:.4f} ({row['R2_Change']:+.4f}, {row['R2_Change_Pct']:+.1f}%)")
        
        # Interpretation (for RMSE, lower is better; for R², higher is better)
        if row['RMSE_Change'] > 0.5:
            print(f"  ⚠️  Significant RMSE increase (worse)")
        elif row['RMSE_Change'] < -0.5:
            print(f"  ✅ Significant RMSE decrease (better)")
        else:
            print(f"  ➖ Minimal RMSE change")
            
        if row['R2_Change'] < -0.1:
            print(f"  ⚠️  Significant R² degradation")
        elif row['R2_Change'] > 0.1:
            print(f"  ✅ Significant R² improvement")
        else:
            print(f"  ➖ Minimal R² change")

# Overall summary
print("\n" + "="*80)
print("OVERALL ASSESSMENT")
print("="*80)

avg_f1_change = clf_df['F1_Change'].mean() if not clf_df.empty else 0
avg_auc_change = clf_df['AUC_Change'].mean() if not clf_df.empty else 0
avg_rmse_change = reg_df['RMSE_Change'].mean() if not reg_df.empty else 0
avg_r2_change = reg_df['R2_Change'].mean() if not reg_df.empty else 0

print(f"\nAverage Changes:")
print(f"  Classification F1:  {avg_f1_change:+.4f} ({(avg_f1_change / clf_df['Baseline_F1'].mean() * 100):+.1f}%)")
print(f"  Classification AUC: {avg_auc_change:+.4f} ({(avg_auc_change / clf_df['Baseline_AUC'].mean() * 100):+.1f}%)")
print(f"  Regression RMSE:    {avg_rmse_change:+.4f} ({(avg_rmse_change / reg_df['Baseline_RMSE'].mean() * 100):+.1f}%)")
print(f"  Regression R²:      {avg_r2_change:+.4f} ({(avg_r2_change / reg_df['Baseline_R2'].mean() * 100):+.1f}%)")

# Key insights
print("\n" + "-"*80)
print("KEY INSIGHTS")
print("-"*80)

insights = []

# Classification insights
if avg_f1_change < -0.1:
    insights.append("❌ Classification models showed significant F1-score degradation with expanded data")
elif avg_f1_change > 0.05:
    insights.append("✅ Classification models improved with expanded data")
else:
    insights.append("➖ Classification models showed mixed or minimal changes")

if avg_auc_change < -0.05:
    insights.append("❌ ROC-AUC scores decreased, indicating worse class separation")
elif avg_auc_change > 0.02:
    insights.append("✅ ROC-AUC scores improved, indicating better class separation")
else:
    insights.append("➖ ROC-AUC scores remained relatively stable")

# Regression insights
if avg_r2_change < -0.2:
    insights.append("❌ Regression models showed significant R² degradation")
elif avg_r2_change > 0.1:
    insights.append("✅ Regression models improved with expanded data")
else:
    insights.append("➖ Regression models showed mixed or minimal changes")

if avg_rmse_change > 1.0:
    insights.append("❌ Prediction errors (RMSE) increased significantly")
elif avg_rmse_change < -0.5:
    insights.append("✅ Prediction errors (RMSE) decreased")
else:
    insights.append("➖ Prediction errors remained relatively stable")

for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")

# Possible explanations
print("\n" + "-"*80)
print("POSSIBLE EXPLANATIONS FOR PERFORMANCE CHANGES")
print("-"*80)

explanations = [
    "1. Data Quality: 2023 transfers may have different characteristics or data quality",
    "2. Feature Reduction: Expanded dataset uses 21 features vs 69 in baseline (70% reduction)",
    "3. Class Imbalance: Adding more data may have changed the class distribution",
    "4. Model Complexity: Models may need retuning for the larger, more diverse dataset",
    "5. Temporal Shift: 2023 season may represent different transfer market dynamics",
    "6. Sample Size: Test set increased from 165 to 297 samples, providing more robust evaluation"
]

for explanation in explanations:
    print(f"  {explanation}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

recommendations = [
    "1. Investigate why feature count dropped from 69 to 21 in expanded dataset",
    "2. Perform feature engineering to restore important predictive features",
    "3. Conduct hyperparameter tuning specifically for the expanded dataset",
    "4. Analyze 2023 transfer data characteristics vs 2020-2022 data",
    "5. Consider ensemble methods that may better handle the expanded data",
    "6. Evaluate if certain subsets (e.g., by league or position) show different trends",
    "7. Implement cross-validation to get more stable performance estimates"
]

for recommendation in recommendations:
    print(f"  {recommendation}")

print("\n" + "="*80)

# Save summary to file
with open('results/performance_comparison_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("MODEL PERFORMANCE COMPARISON: 821 vs 1,483 Records\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset Size:\n")
    f.write(f"  Baseline: {baseline_results['dataset_size']} records\n")
    f.write(f"  Expanded: {expanded_results['dataset_size']} records\n")
    f.write(f"  Increase: +{expanded_results['dataset_size'] - baseline_results['dataset_size']} records\n\n")
    
    f.write("Average Performance Changes:\n")
    f.write(f"  Classification F1:  {avg_f1_change:+.4f}\n")
    f.write(f"  Classification AUC: {avg_auc_change:+.4f}\n")
    f.write(f"  Regression RMSE:    {avg_rmse_change:+.4f}\n")
    f.write(f"  Regression R²:      {avg_r2_change:+.4f}\n\n")
    
    f.write("Key Insights:\n")
    for insight in insights:
        f.write(f"  {insight}\n")
    
    f.write("\nRecommendations:\n")
    for recommendation in recommendations:
        f.write(f"  {recommendation}\n")

print("\n✅ Saved summary to: results/performance_comparison_summary.txt")
print("\n" + "="*80)

