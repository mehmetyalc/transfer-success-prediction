"""
Exploratory Data Analysis (EDA) for Transfer Success Prediction
Analyze integrated transfer and performance data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("TRANSFER SUCCESS PREDICTION - EXPLORATORY DATA ANALYSIS")
print("="*70)

# Load integrated data
df = pd.read_csv('../data/processed/integrated_transfers_performance.csv')

print(f"\n📊 Dataset Overview")
print(f"{'='*70}")
print(f"Total records: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 1. DATA QUALITY ASSESSMENT
# ============================================================================

print(f"\n\n📋 1. DATA QUALITY ASSESSMENT")
print(f"{'='*70}")

print(f"\nMissing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing': missing.values,
    'Percentage': missing_pct.values
})
missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
print(missing_df.to_string(index=False))

# Data completeness by year
print(f"\n\nData Completeness by Year:")
for year in sorted(df['year'].unique()):
    year_df = df[df['year'] == year]
    has_before = year_df['perf_before_goals'].notna().sum()
    has_after = year_df['perf_after_goals'].notna().sum()
    has_both = (year_df['perf_before_goals'].notna() & year_df['perf_after_goals'].notna()).sum()
    
    print(f"\n  {year}:")
    print(f"    Total transfers: {len(year_df):,}")
    print(f"    Has before data: {has_before:,} ({has_before/len(year_df)*100:.1f}%)")
    print(f"    Has after data: {has_after:,} ({has_after/len(year_df)*100:.1f}%)")
    print(f"    Has both: {has_both:,} ({has_both/len(year_df)*100:.1f}%)")

# Filter for complete data (both before and after)
df_complete = df[(df['perf_before_goals'].notna()) & (df['perf_after_goals'].notna())].copy()

print(f"\n\n✅ Complete Dataset (with before & after performance):")
print(f"   Records: {len(df_complete):,} ({len(df_complete)/len(df)*100:.1f}% of total)")

# ============================================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================================

print(f"\n\n📈 2. DESCRIPTIVE STATISTICS")
print(f"{'='*70}")

# Transfer fees
print(f"\nTransfer Fees (EUR millions):")
fee_stats = df_complete['fee_cleaned'].describe()
print(fee_stats.to_string())
print(f"  Transfers with fee data: {df_complete['fee_cleaned'].notna().sum():,} ({df_complete['fee_cleaned'].notna().sum()/len(df_complete)*100:.1f}%)")
print(f"  Free transfers: {(df_complete['fee_cleaned'] == 0).sum():,}")

# Age distribution
print(f"\nAge Distribution:")
age_stats = df_complete['age'].describe()
print(age_stats.to_string())

# Performance metrics
print(f"\nPerformance Metrics:")
perf_cols = ['perf_before_goals', 'perf_after_goals', 'perf_before_assists', 'perf_after_assists']
perf_stats = df_complete[perf_cols].describe()
print(perf_stats.to_string())

# ============================================================================
# 3. TRANSFER DISTRIBUTION ANALYSIS
# ============================================================================

print(f"\n\n🌍 3. TRANSFER DISTRIBUTION ANALYSIS")
print(f"{'='*70}")

# By league
print(f"\nTransfers by League:")
league_counts = df_complete['league_name'].value_counts()
print(league_counts.to_string())

# By position
print(f"\nTransfers by Position:")
position_counts = df_complete['position'].value_counts().head(10)
print(position_counts.to_string())

# Top clubs
print(f"\nTop 10 Clubs by Transfer Activity:")
club_counts = df_complete['club_name'].value_counts().head(10)
print(club_counts.to_string())

# ============================================================================
# 4. PERFORMANCE CHANGE ANALYSIS
# ============================================================================

print(f"\n\n⚡ 4. PERFORMANCE CHANGE ANALYSIS")
print(f"{'='*70}")

# Calculate performance changes
df_complete['goal_change'] = df_complete['perf_after_goals'] - df_complete['perf_before_goals']
df_complete['assist_change'] = df_complete['perf_after_assists'] - df_complete['perf_before_assists']
df_complete['minutes_change'] = df_complete['perf_after_minutes'] - df_complete['perf_before_minutes']

print(f"\nGoal Change Statistics:")
goal_change_stats = df_complete['goal_change'].describe()
print(goal_change_stats.to_string())

print(f"\nPerformance Change Distribution:")
print(f"  Improved (goals+): {(df_complete['goal_change'] > 0).sum():,} ({(df_complete['goal_change'] > 0).sum()/len(df_complete)*100:.1f}%)")
print(f"  Declined (goals-): {(df_complete['goal_change'] < 0).sum():,} ({(df_complete['goal_change'] < 0).sum()/len(df_complete)*100:.1f}%)")
print(f"  Unchanged: {(df_complete['goal_change'] == 0).sum():,} ({(df_complete['goal_change'] == 0).sum()/len(df_complete)*100:.1f}%)")

# Top performers
print(f"\nTop 10 Improved Players (Goal increase):")
top_improved = df_complete.nlargest(10, 'goal_change')[['player_name', 'club_name', 'fee_cleaned', 'perf_before_goals', 'perf_after_goals', 'goal_change']]
print(top_improved.to_string(index=False))

print(f"\nTop 10 Declined Players (Goal decrease):")
top_declined = df_complete.nsmallest(10, 'goal_change')[['player_name', 'club_name', 'fee_cleaned', 'perf_before_goals', 'perf_after_goals', 'goal_change']]
print(top_declined.to_string(index=False))

# ============================================================================
# 5. TRANSFER FEE vs PERFORMANCE ANALYSIS
# ============================================================================

print(f"\n\n💰 5. TRANSFER FEE vs PERFORMANCE ANALYSIS")
print(f"{'='*70}")

df_with_fee = df_complete[df_complete['fee_cleaned'].notna()].copy()

print(f"\nRecords with fee data: {len(df_with_fee):,}")

# Correlation between fee and performance
print(f"\nCorrelation between Transfer Fee and Performance:")
correlations = {
    'Fee vs Before Goals': df_with_fee[['fee_cleaned', 'perf_before_goals']].corr().iloc[0, 1],
    'Fee vs After Goals': df_with_fee[['fee_cleaned', 'perf_after_goals']].corr().iloc[0, 1],
    'Fee vs Goal Change': df_with_fee[['fee_cleaned', 'goal_change']].corr().iloc[0, 1],
}

for key, value in correlations.items():
    print(f"  {key}: {value:.3f}")

# Fee categories
df_with_fee['fee_category'] = pd.cut(df_with_fee['fee_cleaned'], 
                                      bins=[0, 5, 15, 30, 100],
                                      labels=['<5M', '5-15M', '15-30M', '>30M'])

print(f"\nPerformance by Fee Category:")
fee_perf = df_with_fee.groupby('fee_category').agg({
    'perf_before_goals': 'mean',
    'perf_after_goals': 'mean',
    'goal_change': 'mean',
    'player_name': 'count'
}).round(2)
fee_perf.columns = ['Avg Before Goals', 'Avg After Goals', 'Avg Goal Change', 'Count']
print(fee_perf.to_string())

# ============================================================================
# 6. AGE vs PERFORMANCE ANALYSIS
# ============================================================================

print(f"\n\n👤 6. AGE vs PERFORMANCE ANALYSIS")
print(f"{'='*70}")

# Age categories
df_complete['age_category'] = pd.cut(df_complete['age'], 
                                      bins=[0, 21, 25, 29, 100],
                                      labels=['<21', '21-25', '26-29', '30+'])

print(f"\nPerformance by Age Category:")
age_perf = df_complete.groupby('age_category').agg({
    'perf_before_goals': 'mean',
    'perf_after_goals': 'mean',
    'goal_change': 'mean',
    'player_name': 'count'
}).round(2)
age_perf.columns = ['Avg Before Goals', 'Avg After Goals', 'Avg Goal Change', 'Count']
print(age_perf.to_string())

# ============================================================================
# 7. POSITION-BASED ANALYSIS
# ============================================================================

print(f"\n\n⚽ 7. POSITION-BASED ANALYSIS")
print(f"{'='*70}")

# Group similar positions
position_map = {
    'Centre-Forward': 'Forward',
    'Left Winger': 'Forward',
    'Right Winger': 'Forward',
    'Second Striker': 'Forward',
    'Attacking Midfield': 'Midfielder',
    'Central Midfield': 'Midfielder',
    'Defensive Midfield': 'Midfielder',
    'Left Midfield': 'Midfielder',
    'Right Midfield': 'Midfielder',
    'Left-Back': 'Defender',
    'Right-Back': 'Defender',
    'Centre-Back': 'Defender',
    'Goalkeeper': 'Goalkeeper'
}

df_complete['position_group'] = df_complete['position'].map(position_map).fillna('Other')

print(f"\nPerformance by Position Group:")
pos_perf = df_complete.groupby('position_group').agg({
    'perf_before_goals': 'mean',
    'perf_after_goals': 'mean',
    'goal_change': 'mean',
    'player_name': 'count'
}).round(2)
pos_perf.columns = ['Avg Before Goals', 'Avg After Goals', 'Avg Goal Change', 'Count']
print(pos_perf.to_string())

# ============================================================================
# 8. SUCCESS DEFINITION
# ============================================================================

print(f"\n\n🎯 8. TRANSFER SUCCESS DEFINITION")
print(f"{'='*70}")

# Define success based on multiple criteria
df_complete['success_goals'] = (df_complete['goal_change'] > 0).astype(int)
df_complete['success_minutes'] = (df_complete['minutes_change'] > 0).astype(int)
df_complete['success_composite'] = ((df_complete['goal_change'] >= 0) & 
                                    (df_complete['minutes_change'] > 500)).astype(int)

print(f"\nSuccess Rate by Different Definitions:")
print(f"  Goal Improvement: {df_complete['success_goals'].mean()*100:.1f}%")
print(f"  Minutes Increase: {df_complete['success_minutes'].mean()*100:.1f}%")
print(f"  Composite (Goals stable + Minutes >500): {df_complete['success_composite'].mean()*100:.1f}%")

# ============================================================================
# 9. SAVE PROCESSED DATA
# ============================================================================

print(f"\n\n💾 9. SAVING PROCESSED DATA")
print(f"{'='*70}")

# Save complete dataset with new features
output_file = '../data/processed/transfers_with_features.csv'
df_complete.to_csv(output_file, index=False)
print(f"✅ Saved processed data to: {output_file}")
print(f"   Records: {len(df_complete):,}")
print(f"   Columns: {len(df_complete.columns)}")

# Summary statistics
summary_file = '../data/processed/eda_summary.txt'
with open(summary_file, 'w') as f:
    f.write("TRANSFER SUCCESS PREDICTION - EDA SUMMARY\n")
    f.write("="*70 + "\n\n")
    f.write(f"Total Records: {len(df_complete):,}\n")
    f.write(f"Date Range: 2022\n")
    f.write(f"Leagues: {df_complete['league_name'].nunique()}\n")
    f.write(f"Clubs: {df_complete['club_name'].nunique()}\n")
    f.write(f"Players: {df_complete['player_name'].nunique()}\n\n")
    f.write(f"Performance Change:\n")
    f.write(f"  Improved: {(df_complete['goal_change'] > 0).sum():,} ({(df_complete['goal_change'] > 0).mean()*100:.1f}%)\n")
    f.write(f"  Declined: {(df_complete['goal_change'] < 0).sum():,} ({(df_complete['goal_change'] < 0).mean()*100:.1f}%)\n")
    f.write(f"  Unchanged: {(df_complete['goal_change'] == 0).sum():,} ({(df_complete['goal_change'] == 0).mean()*100:.1f}%)\n")

print(f"✅ Saved summary to: {summary_file}")

print(f"\n\n{'='*70}")
print(f"EDA COMPLETE!")
print(f"{'='*70}")
print(f"\nKey Findings:")
print(f"  • {len(df_complete):,} transfers with complete performance data")
print(f"  • {(df_complete['goal_change'] > 0).mean()*100:.1f}% of transfers showed goal improvement")
print(f"  • Average goal change: {df_complete['goal_change'].mean():.2f}")
print(f"  • Fee correlation with goal change: {df_with_fee[['fee_cleaned', 'goal_change']].corr().iloc[0, 1]:.3f}")
print(f"\nNext Steps:")
print(f"  1. Create visualizations")
print(f"  2. Feature engineering")
print(f"  3. Model development")

