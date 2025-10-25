"""
Integrate 2023 Davidcariboo transfers with FBref performance data
Before: 2022-23, After: 2023-24
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


logger.info("="*70)
logger.info("INTEGRATING 2023 TRANSFERS")
logger.info("="*70)

# Load Davidcariboo data
logger.info("\nLoading Davidcariboo data...")
df_transfers = pd.read_csv('data/external/davidcariboo/transfers.csv')
df_players = pd.read_csv('data/external/davidcariboo/players.csv')
df_clubs = pd.read_csv('data/external/davidcariboo/clubs.csv')

# Parse dates
df_transfers['transfer_date'] = pd.to_datetime(df_transfers['transfer_date'], dayfirst=True)
df_transfers['year'] = df_transfers['transfer_date'].dt.year

# Filter 2023
df_2023 = df_transfers[df_transfers['year'] == 2023].copy()
logger.info(f"2023 transfers: {len(df_2023)}")

# Merge with players
df_2023 = df_2023.merge(
    df_players[['player_id', 'name', 'position', 'date_of_birth']],
    on='player_id',
    how='left'
)

# Merge with clubs to get league info
df_2023 = df_2023.merge(
    df_clubs[['club_id', 'domestic_competition_id']],
    left_on='to_club_id',
    right_on='club_id',
    how='left'
)

# Map competition IDs to leagues
comp_to_league = {
    'GB1': 'Premier League',
    'ES1': 'Primera Division',
    'IT1': 'Serie A',
    'L1': 'Bundesliga',
    'FR1': 'Ligue 1'
}

df_2023['league_name'] = df_2023['domestic_competition_id'].map(comp_to_league)

# Filter Big 5 leagues
df_2023_big5 = df_2023[df_2023['league_name'].notna()].copy()
logger.info(f"Big 5 leagues: {len(df_2023_big5)}")

logger.info("\nTransfers by league:")
for league in df_2023_big5['league_name'].value_counts().items():
    logger.info(f"  {league[0]}: {league[1]}")

# Country mapping for FBref
country_map = {
    'Premier League': 'England',
    'Primera Division': 'Spain',
    'Serie A': 'Italy',
    'Bundesliga': 'Germany',
    'Ligue 1': 'France'
}

# Initialize performance columns
for prefix in ['before', 'after']:
    for metric in ['goals', 'assists', 'minutes', 'matches']:
        df_2023_big5[f'perf_{prefix}_{metric}'] = np.nan

# Load FBref data and match
logger.info("\n" + "="*70)
logger.info("MATCHING WITH FBREF DATA")
logger.info("="*70)

for league, country in country_map.items():
    league_df = df_2023_big5[df_2023_big5['league_name'] == league].copy()
    if len(league_df) == 0:
        continue
    
    logger.info(f"\n{league}: {len(league_df)} transfers")
    
    # Load FBref (2223 for before, 2324 for after)
    for period, season in [('before', '2223'), ('after', '2324')]:
        try:
            df_fbref = pd.read_csv(
                f'data/raw/fbref/{country}_{season}_player_stats_standard.csv',
                header=[0, 1]
            )
            df_fbref.columns = ['_'.join(c).strip() for c in df_fbref.columns]
            
            # Get player column (index 3)
            player_col = df_fbref.columns[3]
            df_fbref['player_norm'] = df_fbref[player_col].apply(normalize_name)
            
            matched = 0
            for idx in league_df.index:
                player_name = league_df.at[idx, 'name']
                player_norm = normalize_name(player_name)
                
                # Fuzzy match
                df_fbref['score'] = df_fbref['player_norm'].apply(
                    lambda x: fuzz.ratio(x, player_norm)
                )
                matches = df_fbref[df_fbref['score'] > 80]
                
                if len(matches) > 0:
                    best = matches.loc[matches['score'].idxmax()]
                    
                    # Extract metrics
                    for c in df_fbref.columns:
                        if c.endswith('_Gls') and 'Performance' in c:
                            df_2023_big5.at[idx, f'perf_{period}_goals'] = best[c]
                        elif c.endswith('_Ast') and 'Performance' in c:
                            df_2023_big5.at[idx, f'perf_{period}_assists'] = best[c]
                        elif c.endswith('_Min') and 'Playing Time' in c:
                            df_2023_big5.at[idx, f'perf_{period}_minutes'] = best[c]
                        elif c.endswith('_MP') and 'Playing Time' in c:
                            df_2023_big5.at[idx, f'perf_{period}_matches'] = best[c]
                    
                    matched += 1
            
            logger.info(f"  {period}: matched {matched} ({matched/len(league_df)*100:.1f}%)")
            
        except Exception as e:
            logger.warning(f"  Error loading {period}: {e}")

# Calculate changes
df_2023_big5['goal_change'] = df_2023_big5['perf_after_goals'] - df_2023_big5['perf_before_goals']
df_2023_big5['assist_change'] = df_2023_big5['perf_after_assists'] - df_2023_big5['perf_before_assists']

# Rename columns to match existing format
df_2023_big5 = df_2023_big5.rename(columns={
    'name': 'player_name',
    'transfer_fee': 'fee',
    'position': 'player_position'
})

# Select relevant columns
cols_to_keep = [
    'player_name', 'league_name', 'year', 'fee', 'market_value_in_eur',
    'player_position', 'from_club_name', 'to_club_name',
    'perf_before_goals', 'perf_before_assists', 'perf_before_minutes', 'perf_before_matches',
    'perf_after_goals', 'perf_after_assists', 'perf_after_minutes', 'perf_after_matches',
    'goal_change', 'assist_change'
]

df_2023_final = df_2023_big5[cols_to_keep].copy()

# Save
output = 'data/processed/transfers_2023_integrated.csv'
df_2023_final.to_csv(output, index=False)
logger.info(f"\n✅ Saved: {output}")

# Summary
logger.info("\n" + "="*70)
logger.info("SUMMARY")
logger.info("="*70)

complete = df_2023_final.dropna(subset=['perf_before_goals', 'perf_after_goals'])
logger.info(f"\nTotal 2023 transfers (Big 5): {len(df_2023_final)}")
logger.info(f"Complete records (before + after): {len(complete)}")
logger.info(f"Completion rate: {len(complete)/len(df_2023_final)*100:.1f}%")

logger.info("\nBy league:")
for league in df_2023_final['league_name'].unique():
    total = len(df_2023_final[df_2023_final['league_name'] == league])
    comp = len(complete[complete['league_name'] == league])
    logger.info(f"  {league}: {total} total, {comp} complete ({comp/total*100:.1f}%)")

logger.info("\n" + "="*70)
logger.info("READY TO MERGE WITH EXISTING DATA!")
logger.info("="*70)

