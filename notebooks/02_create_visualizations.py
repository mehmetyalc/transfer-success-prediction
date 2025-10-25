"""
Create visualizations for Transfer Success Prediction EDA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load processed data
df = pd.read_csv('../data/processed/transfers_with_features.csv')

print("Creating visualizations...")

# Create output directory
import os
os.makedirs('../results/figures', exist_ok=True)

# ============================================================================
# 1. Performance Change Distribution
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Transfer Performance Analysis', fontsize=16, fontweight='bold')

# 1.1 Goal Change Distribution
axes[0, 0].hist(df['goal_change'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
axes[0, 0].set_xlabel('Goal Change (After - Before)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Distribution of Goal Change After Transfer', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 1.2 Before vs After Goals Scatter
axes[0, 1].scatter(df['perf_before_goals'], df['perf_after_goals'], alpha=0.5, s=30)
max_val = max(df['perf_before_goals'].max(), df['perf_after_goals'].max())
axes[0, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='No change line')
axes[0, 1].set_xlabel('Goals Before Transfer', fontsize=12)
axes[0, 1].set_ylabel('Goals After Transfer', fontsize=12)
axes[0, 1].set_title('Goals: Before vs After Transfer', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 1.3 Minutes Change
axes[1, 0].hist(df['minutes_change'], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Minutes Change (After - Before)', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Distribution of Playing Time Change', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 1.4 Success Rate Pie Chart
success_counts = df['success_goals'].value_counts()
colors = ['#ff6b6b', '#51cf66']
labels = ['Declined/Same', 'Improved']
axes[1, 1].pie(success_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
axes[1, 1].set_title('Transfer Success Rate (Goal Improvement)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/figures/01_performance_change_overview.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 01_performance_change_overview.png")
plt.close()

# ============================================================================
# 2. Fee vs Performance
# ============================================================================

df_with_fee = df[df['fee_cleaned'].notna()].copy()
df_with_fee['fee_category'] = pd.cut(df_with_fee['fee_cleaned'], 
                                      bins=[0, 5, 15, 30, 100],
                                      labels=['<5M', '5-15M', '15-30M', '>30M'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Transfer Fee vs Performance Analysis', fontsize=16, fontweight='bold')

# 2.1 Fee vs Goal Change
axes[0].scatter(df_with_fee['fee_cleaned'], df_with_fee['goal_change'], alpha=0.6, s=50)
axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Transfer Fee (EUR millions)', fontsize=12)
axes[0].set_ylabel('Goal Change', fontsize=12)
axes[0].set_title(f'Fee vs Goal Change\n(Correlation: {df_with_fee[["fee_cleaned", "goal_change"]].corr().iloc[0,1]:.3f})', 
                  fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 2.2 Fee Category Performance
fee_perf = df_with_fee.groupby('fee_category').agg({
    'goal_change': 'mean',
    'player_name': 'count'
}).reset_index()
fee_perf.columns = ['Fee Category', 'Avg Goal Change', 'Count']

x = range(len(fee_perf))
bars = axes[1].bar(x, fee_perf['Avg Goal Change'], color=['green' if v > 0 else 'red' for v in fee_perf['Avg Goal Change']])
axes[1].set_xticks(x)
axes[1].set_xticklabels(fee_perf['Fee Category'])
axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1].set_xlabel('Fee Category', fontsize=12)
axes[1].set_ylabel('Average Goal Change', fontsize=12)
axes[1].set_title('Average Goal Change by Fee Category', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Add count labels
for i, (bar, count) in enumerate(zip(bars, fee_perf['Count'])):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'n={count}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

# 2.3 Fee Distribution
axes[2].hist(df_with_fee['fee_cleaned'], bins=30, edgecolor='black', alpha=0.7, color='purple')
axes[2].set_xlabel('Transfer Fee (EUR millions)', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].set_title(f'Transfer Fee Distribution\n(Mean: €{df_with_fee["fee_cleaned"].mean():.1f}M, Median: €{df_with_fee["fee_cleaned"].median():.1f}M)', 
                  fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/figures/02_fee_vs_performance.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 02_fee_vs_performance.png")
plt.close()

# ============================================================================
# 3. Age and Position Analysis
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Age and Position Impact on Transfer Success', fontsize=16, fontweight='bold')

# 3.1 Age Category Performance
age_perf = df.groupby('age_category').agg({
    'goal_change': 'mean',
    'player_name': 'count'
}).reset_index()
age_perf.columns = ['Age Category', 'Avg Goal Change', 'Count']

x = range(len(age_perf))
bars = axes[0].bar(x, age_perf['Avg Goal Change'], color=['green' if v > 0 else 'red' for v in age_perf['Avg Goal Change']])
axes[0].set_xticks(x)
axes[0].set_xticklabels(age_perf['Age Category'])
axes[0].axhline(0, color='black', linestyle='-', linewidth=1)
axes[0].set_xlabel('Age Category', fontsize=12)
axes[0].set_ylabel('Average Goal Change', fontsize=12)
axes[0].set_title('Average Goal Change by Age Category', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

for i, (bar, count) in enumerate(zip(bars, age_perf['Count'])):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'n={count}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

# 3.2 Position Group Performance
pos_perf = df.groupby('position_group').agg({
    'goal_change': 'mean',
    'player_name': 'count'
}).reset_index()
pos_perf = pos_perf[pos_perf['position_group'] != 'Other']  # Remove 'Other' with only 2 samples
pos_perf.columns = ['Position', 'Avg Goal Change', 'Count']
pos_perf = pos_perf.sort_values('Avg Goal Change', ascending=False)

x = range(len(pos_perf))
bars = axes[1].barh(x, pos_perf['Avg Goal Change'], color=['green' if v > 0 else 'red' for v in pos_perf['Avg Goal Change']])
axes[1].set_yticks(x)
axes[1].set_yticklabels(pos_perf['Position'])
axes[1].axvline(0, color='black', linestyle='-', linewidth=1)
axes[1].set_xlabel('Average Goal Change', fontsize=12)
axes[1].set_ylabel('Position Group', fontsize=12)
axes[1].set_title('Average Goal Change by Position', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

for i, (bar, count) in enumerate(zip(bars, pos_perf['Count'])):
    width = bar.get_width()
    axes[1].text(width, bar.get_y() + bar.get_height()/2.,
                f' n={count}', ha='left' if width > 0 else 'right', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('../results/figures/03_age_position_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 03_age_position_analysis.png")
plt.close()

# ============================================================================
# 4. League Comparison
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('League-wise Transfer Performance', fontsize=16, fontweight='bold')

# 4.1 Transfer count by league
league_counts = df['league_name'].value_counts()
axes[0].barh(range(len(league_counts)), league_counts.values, color='steelblue')
axes[0].set_yticks(range(len(league_counts)))
axes[0].set_yticklabels(league_counts.index)
axes[0].set_xlabel('Number of Transfers', fontsize=12)
axes[0].set_title('Transfer Activity by League', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

for i, v in enumerate(league_counts.values):
    axes[0].text(v, i, f' {v}', va='center', fontsize=10)

# 4.2 Average goal change by league
league_perf = df.groupby('league_name').agg({
    'goal_change': 'mean',
    'player_name': 'count'
}).reset_index()
league_perf.columns = ['League', 'Avg Goal Change', 'Count']
league_perf = league_perf.sort_values('Avg Goal Change', ascending=False)

x = range(len(league_perf))
bars = axes[1].barh(x, league_perf['Avg Goal Change'], 
                    color=['green' if v > 0 else 'red' for v in league_perf['Avg Goal Change']])
axes[1].set_yticks(x)
axes[1].set_yticklabels(league_perf['League'])
axes[1].axvline(0, color='black', linestyle='-', linewidth=1)
axes[1].set_xlabel('Average Goal Change', fontsize=12)
axes[1].set_title('Average Goal Change by League', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../results/figures/04_league_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 04_league_comparison.png")
plt.close()

# ============================================================================
# 5. Correlation Heatmap
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

# Select numeric columns
numeric_cols = ['age', 'fee_cleaned', 'perf_before_goals', 'perf_after_goals', 
                'perf_before_assists', 'perf_after_assists',
                'perf_before_minutes', 'perf_after_minutes',
                'goal_change', 'assist_change', 'minutes_change']

corr_matrix = df[numeric_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix of Transfer and Performance Metrics', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('../results/figures/05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 05_correlation_heatmap.png")
plt.close()

print("\n✅ All visualizations created successfully!")
print(f"   Saved to: ../results/figures/")

