"""
Enrich ML-ready data with advanced metrics from FBref
xG, xAG, Passing Stats, Progressive Actions
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import unicodedata
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_name(name):
    """Normalize player name"""
    if pd.isna(name):
        return ""
    name = str(name)
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    return name.lower().strip()


def load_fbref_data(country, season, data_type):
    """Load FBref data"""
    try:
        filename = f'data/raw/fbref/{country}_{season}_player_stats_{data_type}.csv'
        df = pd.read_csv(filename, header=[0, 1])
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        return df
    except Exception as e:
        logger.warning(f"Could not load {filename}: {e}")
        return pd.DataFrame()


def enrich_ml_data():
    """Add advanced metrics to ML-ready data"""
    
    logger.info("="*70)
    logger.info("ENRICHING ML DATA WITH ADVANCED METRICS")
    logger.info("="*70)
    
    # Load ML-ready data
    df_ml = pd.read_csv('data/processed/transfers_ml_ready.csv')
    logger.info(f"\nLoaded {len(df_ml)} ML-ready records")
    
    # Load original integrated data to get player names and leagues
    df_integrated = pd.read_csv('data/processed/integrated_transfers_performance.csv')
    
    # Merge to get player info
    df_ml = df_ml.merge(
        df_integrated[['player_name', 'league_name', 'year']],
        left_index=True,
        right_index=True,
        how='left'
    )
    
    # Country mapping
    country_map = {
        'Premier League': 'England',
        'Primera Division': 'Spain',
        'Serie A': 'Italy',
        '1 Bundesliga': 'Germany',
        'Ligue 1': 'France'
    }
    
    # All records are from 2022, so:
    # - Before: 2021-22 (2122)
    # - After: 2022-23 (2223)
    
    # Initialize new columns
    new_metrics = [
        'xG', 'xAG', 'npxG', 'shots', 'shots_on_target',
        'pass_completion_pct', 'progressive_passes', 'progressive_carries',
        'key_passes', 'passes_into_final_third'
    ]
    
    for prefix in ['before', 'after']:
        for metric in new_metrics:
            df_ml[f'{prefix}_{metric}'] = np.nan
    
    # Process each league
    for league, country in country_map.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {league}")
        logger.info(f"{'='*70}")
        
        league_data = df_ml[df_ml['league_name'] == league].copy()
        if len(league_data) == 0:
            continue
        
        logger.info(f"Found {len(league_data)} transfers")
        
        # Load FBref data for before (2122) and after (2223)
        for period, season in [('before', '2122'), ('after', '2223')]:
            logger.info(f"\nLoading {period} data ({season})...")
            
            # Load shooting stats
            df_shooting = load_fbref_data(country, season, 'shooting')
            # Load passing stats
            df_passing = load_fbref_data(country, season, 'passing')
            
            if df_shooting.empty and df_passing.empty:
                logger.warning(f"No data for {league} {season}")
                continue
            
            # Prepare player names
            if not df_shooting.empty:
                player_col_shooting = [c for c in df_shooting.columns if 'player' in c.lower()][0]
                df_shooting['player_norm'] = df_shooting[player_col_shooting].apply(normalize_name)
            
            if not df_passing.empty:
                player_col_passing = [c for c in df_passing.columns if 'player' in c.lower()][0]
                df_passing['player_norm'] = df_passing[player_col_passing].apply(normalize_name)
            
            # Match each player
            matched = 0
            for idx in league_data.index:
                player_name = league_data.at[idx, 'player_name']
                player_norm = normalize_name(player_name)
                
                # Match in shooting data
                if not df_shooting.empty:
                    df_shooting['match_score'] = df_shooting['player_norm'].apply(
                        lambda x: fuzz.ratio(x, player_norm)
                    )
                    matches = df_shooting[df_shooting['match_score'] > 80]
                    
                    if len(matches) > 0:
                        best = matches.loc[matches['match_score'].idxmax()]
                        
                        # Extract metrics
                        for col in df_shooting.columns:
                            if 'Expected_xG' in col and 'xAG' not in col:
                                df_ml.at[idx, f'{period}_xG'] = best[col]
                            elif 'Expected_xAG' in col or 'Expected_xA' in col:
                                df_ml.at[idx, f'{period}_xAG'] = best[col]
                            elif 'Expected_npxG' in col:
                                df_ml.at[idx, f'{period}_npxG'] = best[col]
                            elif col.endswith('_Sh') or 'Standard_Sh' in col:
                                df_ml.at[idx, f'{period}_shots'] = best[col]
                            elif col.endswith('_SoT') or 'Standard_SoT' in col:
                                df_ml.at[idx, f'{period}_shots_on_target'] = best[col]
                
                # Match in passing data
                if not df_passing.empty:
                    df_passing['match_score'] = df_passing['player_norm'].apply(
                        lambda x: fuzz.ratio(x, player_norm)
                    )
                    matches = df_passing[df_passing['match_score'] > 80]
                    
                    if len(matches) > 0:
                        best = matches.loc[matches['match_score'].idxmax()]
                        
                        # Extract metrics
                        for col in df_passing.columns:
                            if 'Cmp%' in col and 'Total' in col:
                                df_ml.at[idx, f'{period}_pass_completion_pct'] = best[col]
                            elif 'PrgP' in col:
                                df_ml.at[idx, f'{period}_progressive_passes'] = best[col]
                            elif 'PrgC' in col:
                                df_ml.at[idx, f'{period}_progressive_carries'] = best[col]
                            elif 'KP' in col:
                                df_ml.at[idx, f'{period}_key_passes'] = best[col]
                            elif '1/3' in col:
                                df_ml.at[idx, f'{period}_passes_into_final_third'] = best[col]
                        
                        matched += 1
            
            logger.info(f"✅ Matched {matched} players")
    
    # Drop temporary columns
    df_ml = df_ml.drop(['player_name', 'league_name', 'year'], axis=1, errors='ignore')
    
    # Save enriched data
    output_file = 'data/processed/transfers_ml_enriched.csv'
    df_ml.to_csv(output_file, index=False)
    logger.info(f"\n✅ Saved: {output_file}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    logger.info(f"\nOriginal features: {len(df_ml.columns) - len(new_metrics)*2}")
    logger.info(f"New metrics added: {len(new_metrics)*2}")
    logger.info(f"Total features: {len(df_ml.columns)}")
    
    logger.info("\nData completeness:")
    for metric in ['xG', 'xAG', 'pass_completion_pct', 'progressive_passes']:
        before_count = df_ml[f'before_{metric}'].notna().sum()
        after_count = df_ml[f'after_{metric}'].notna().sum()
        logger.info(f"  {metric}:")
        logger.info(f"    Before: {before_count} ({before_count/len(df_ml)*100:.1f}%)")
        logger.info(f"    After: {after_count} ({after_count/len(df_ml)*100:.1f}%)")
    
    logger.info("\n" + "="*70)
    logger.info("ENRICHMENT COMPLETE!")
    logger.info("="*70)


if __name__ == "__main__":
    enrich_ml_data()

