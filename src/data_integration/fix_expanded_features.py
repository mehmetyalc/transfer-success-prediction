"""
Fix expanded dataset by adding missing engineered features
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("FIXING EXPANDED DATASET - ADDING MISSING FEATURES")
logger.info("="*80)

# Load expanded dataset
df = pd.read_csv('data/processed/transfers_ml_ready_expanded.csv')
logger.info(f"\nLoaded {len(df)} records with {len(df.columns)} features")

# Display current columns
logger.info(f"\nCurrent features: {sorted(df.columns.tolist())[:20]}...")

# Add missing engineered features

# 1. Age categories
logger.info("\n1. Adding age categories...")
df['age_category'] = pd.cut(df['age'], bins=[0, 21, 26, 30, 100], 
                             labels=['young', 'prime', 'veteran', 'senior'])

# 2. Position groups
logger.info("2. Adding position groups...")
if 'is_forward' in df.columns:
    df['position_group'] = 'Unknown'
    df.loc[df['is_forward'] == 1, 'position_group'] = 'Forward'
    df.loc[df['is_midfielder'] == 1, 'position_group'] = 'Midfielder'
    df.loc[df['is_defender'] == 1, 'position_group'] = 'Defender'
    df.loc[df['is_goalkeeper'] == 1, 'position_group'] = 'Goalkeeper'

# 3. Fee features
logger.info("3. Adding fee features...")
df['has_fee'] = (df['fee_millions'] > 0).astype(int)
df['fee_log'] = np.log1p(df['fee_millions'])

# 4. Performance changes
logger.info("4. Adding performance change metrics...")
if 'goals_per_90_before' in df.columns and 'goals_per_90_after' in df.columns:
    df['goals_change'] = df['goals_per_90_after'] - df['goals_per_90_before']
    df['assists_change'] = df['assists_per_90_after'] - df['assists_per_90_before']
    df['goal_contribution_change'] = df['goal_contribution_after'] - df['goal_contribution_before']

if 'perf_before_minutes' in df.columns and 'perf_after_minutes' in df.columns:
    df['minutes_change'] = df['perf_after_minutes'] - df['perf_before_minutes']

# 5. Minutes per match
logger.info("5. Adding minutes per match...")
if 'perf_before_minutes' in df.columns and 'perf_before_matches' in df.columns:
    df['minutes_per_match_before'] = df['perf_before_minutes'] / df['perf_before_matches'].replace(0, 1)
    df['minutes_per_match_after'] = df['perf_after_minutes'] / df['perf_after_matches'].replace(0, 1)

# 6. League and position averages
logger.info("6. Computing league and position averages...")

# Get league information from league dummies
league_cols = [c for c in df.columns if c.startswith('league_')]
if league_cols:
    # Create league_name from dummies
    df['league_name'] = 'Unknown'
    for col in league_cols:
        league = col.replace('league_', '')
        df.loc[df[col] == 1, 'league_name'] = league

# Compute averages by league
if 'league_name' in df.columns:
    league_stats = df.groupby('league_name').agg({
        'goals_per_90_before': 'mean',
        'assists_per_90_before': 'mean',
        'perf_before_minutes': 'mean'
    }).add_suffix('_league_avg')
    
    df = df.merge(league_stats, left_on='league_name', right_index=True, how='left')
    
    # Compute differences from league average
    df['goals_vs_league_avg'] = df['goals_per_90_before'] - df['goals_per_90_before_league_avg']
    df['assists_vs_league_avg'] = df['assists_per_90_before'] - df['assists_per_90_before_league_avg']

# Compute averages by position
if 'position_group' in df.columns:
    position_stats = df.groupby('position_group').agg({
        'goals_per_90_before': 'mean',
        'assists_per_90_before': 'mean',
        'perf_before_minutes': 'mean'
    }).add_suffix('_position_avg')
    
    df = df.merge(position_stats, left_on='position_group', right_index=True, how='left')
    
    # Compute differences from position average
    df['goals_vs_position_avg'] = df['goals_per_90_before'] - df['goals_per_90_before_position_avg']
    df['assists_vs_position_avg'] = df['assists_per_90_before'] - df['assists_per_90_before_position_avg']
    
    # Rename to match baseline naming
    df['perf_before_goals_position_avg'] = df['goals_per_90_before_position_avg']
    df['perf_before_assists_position_avg'] = df['assists_per_90_before_position_avg']
    df['perf_before_minutes_position_avg'] = df['perf_before_minutes_position_avg']
    
    df['perf_before_goals_league_avg'] = df['goals_per_90_before_league_avg']
    df['perf_before_assists_league_avg'] = df['assists_per_90_before_league_avg']

# 7. Success metrics
logger.info("7. Adding success metrics...")
if 'target_success_goals' in df.columns and 'target_success_minutes' in df.columns:
    df['success_composite'] = ((df['target_success_goals'] == 1) & 
                               (df['target_success_minutes'] == 1)).astype(int)

# 8. Fill any NaN values
logger.info("8. Handling missing values...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# Save fixed dataset
output_path = 'data/processed/transfers_ml_ready_expanded_fixed.csv'
df.to_csv(output_path, index=False)

logger.info(f"\n✅ Fixed dataset saved to: {output_path}")
logger.info(f"   Records: {len(df)}")
logger.info(f"   Features: {len(df.columns)}")

# Compare with baseline
baseline = pd.read_csv('data/processed/transfers_ml_ready.csv')
baseline_cols = set(baseline.columns)
expanded_cols = set(df.columns)

missing = baseline_cols - expanded_cols
extra = expanded_cols - baseline_cols

logger.info(f"\n" + "="*80)
logger.info("FEATURE COMPARISON WITH BASELINE")
logger.info("="*80)
logger.info(f"\nBaseline features: {len(baseline_cols)}")
logger.info(f"Expanded features: {len(expanded_cols)}")
logger.info(f"\nStill missing from baseline: {len(missing)}")
if missing:
    for col in sorted(missing)[:10]:
        logger.info(f"  - {col}")
    if len(missing) > 10:
        logger.info(f"  ... and {len(missing)-10} more")

logger.info(f"\nExtra in expanded: {len(extra)}")
if extra:
    for col in sorted(extra)[:10]:
        logger.info(f"  + {col}")
    if len(extra) > 10:
        logger.info(f"  ... and {len(extra)-10} more")

logger.info("\n" + "="*80)
logger.info("READY FOR MODEL RETRAINING!")
logger.info("="*80)

