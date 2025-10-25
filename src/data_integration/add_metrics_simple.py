"""
Simple script to add advanced metrics to integrated transfers
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import unicodedata
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_name(name):
    if pd.isna(name):
        return ""
    name = str(name)
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    return name.lower().strip()


logger.info("Loading integrated transfers...")
df = pd.read_csv('data/processed/integrated_transfers_performance.csv')
logger.info(f"Loaded {len(df)} transfers")

# Filter to complete records only (2022 transfers)
df_complete = df.dropna(subset=['perf_before_goals', 'perf_after_goals']).copy()
logger.info(f"Complete records: {len(df_complete)}")

# Country mapping
country_map = {
    'Premier League': 'England',
    'Primera Division': 'Spain',
    'Serie A': 'Italy',
    '1 Bundesliga': 'Germany',
    'Ligue 1': 'France'
}

# Initialize new columns
new_cols = ['xG', 'xAG', 'npxG', 'shots', 'pass_cmp_pct', 'prog_passes', 'prog_carries', 'key_passes']
for prefix in ['before', 'after']:
    for col in new_cols:
        df_complete[f'{prefix}_{col}'] = np.nan

# Process each league
for league, country in country_map.items():
    logger.info(f"\nProcessing {league}...")
    
    league_df = df_complete[df_complete['league_name'] == league].copy()
    if len(league_df) == 0:
        continue
    
    logger.info(f"  {len(league_df)} transfers")
    
    # Load FBref data (2122 for before, 2223 for after)
    for period, season in [('before', '2122'), ('after', '2223')]:
        try:
            # Load shooting
            df_shoot = pd.read_csv(f'data/raw/fbref/{country}_{season}_player_stats_shooting.csv', header=[0,1])
            df_shoot.columns = ['_'.join(c).strip() for c in df_shoot.columns]
            
            # Load passing  
            df_pass = pd.read_csv(f'data/raw/fbref/{country}_{season}_player_stats_passing.csv', header=[0,1])
            df_pass.columns = ['_'.join(c).strip() for c in df_pass.columns]
            
            # Get player columns (it's in the first row, column 3)
            # Use column 3 directly
            player_col_shoot = df_shoot.columns[3]
            player_col_pass = df_pass.columns[3]
            
            df_shoot['player_norm'] = df_shoot[player_col_shoot].apply(normalize_name)
            df_pass['player_norm'] = df_pass[player_col_pass].apply(normalize_name)
            
            matched = 0
            for idx in league_df.index:
                player_norm = normalize_name(league_df.at[idx, 'player_name'])
                
                # Match in shooting
                df_shoot['score'] = df_shoot['player_norm'].apply(lambda x: fuzz.ratio(x, player_norm))
                shoot_match = df_shoot[df_shoot['score'] > 80]
                
                if len(shoot_match) > 0:
                    best = shoot_match.loc[shoot_match['score'].idxmax()]
                    
                    # Extract shooting metrics
                    for c in df_shoot.columns:
                        if 'Expected_xG' in c and 'xAG' not in c and 'npxG' not in c:
                            df_complete.at[idx, f'{period}_xG'] = best[c]
                        elif 'Expected_xAG' in c or 'Expected_xA' in c:
                            df_complete.at[idx, f'{period}_xAG'] = best[c]
                        elif 'Expected_npxG' in c:
                            df_complete.at[idx, f'{period}_npxG'] = best[c]
                        elif c.endswith('_Sh') and 'Standard' in c:
                            df_complete.at[idx, f'{period}_shots'] = best[c]
                
                # Match in passing
                df_pass['score'] = df_pass['player_norm'].apply(lambda x: fuzz.ratio(x, player_norm))
                pass_match = df_pass[df_pass['score'] > 80]
                
                if len(pass_match) > 0:
                    best = pass_match.loc[pass_match['score'].idxmax()]
                    
                    # Extract passing metrics
                    for c in df_pass.columns:
                        if 'Cmp%' in c and 'Total' in c:
                            df_complete.at[idx, f'{period}_pass_cmp_pct'] = best[c]
                        elif 'PrgP' in c:
                            df_complete.at[idx, f'{period}_prog_passes'] = best[c]
                        elif 'PrgC' in c:
                            df_complete.at[idx, f'{period}_prog_carries'] = best[c]
                        elif 'KP' in c:
                            df_complete.at[idx, f'{period}_key_passes'] = best[c]
                    
                    matched += 1
            
            logger.info(f"    {period}: matched {matched}")
            
        except Exception as e:
            logger.warning(f"    Error loading {period}: {e}")

# Save
output = 'data/processed/transfers_with_advanced_metrics.csv'
df_complete.to_csv(output, index=False)
logger.info(f"\n✅ Saved: {output}")

# Summary
logger.info("\nMetric completeness:")
for metric in ['xG', 'xAG', 'pass_cmp_pct', 'prog_passes']:
    before = df_complete[f'before_{metric}'].notna().sum()
    after = df_complete[f'after_{metric}'].notna().sum()
    logger.info(f"  {metric}: before={before} ({before/len(df_complete)*100:.1f}%), after={after} ({after/len(df_complete)*100:.1f}%)")

logger.info(f"\nTotal features: {len(df_complete.columns)}")
logger.info("Done!")

