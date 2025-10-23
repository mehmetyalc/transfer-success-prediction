"""
Data Validation and Summary Script
Validates collected data and generates summary statistics
"""

import pandas as pd
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = 'data/raw/fbref'

def analyze_collected_data():
    """Analyze all collected data files"""
    logger.info("=" * 60)
    logger.info("DATA VALIDATION AND SUMMARY")
    logger.info("=" * 60)
    
    # Get all CSV files
    csv_files = glob.glob(f"{DATA_DIR}/*.csv")
    csv_files = [f for f in csv_files if 'test' not in f and 'log' not in f]
    
    logger.info(f"\nTotal files collected: {len(csv_files)}")
    
    # Categorize files
    player_stats_files = [f for f in csv_files if 'player_stats' in f]
    team_stats_files = [f for f in csv_files if 'team_stats' in f]
    schedule_files = [f for f in csv_files if 'schedule' in f]
    
    logger.info(f"  - Player stats files: {len(player_stats_files)}")
    logger.info(f"  - Team stats files: {len(team_stats_files)}")
    logger.info(f"  - Schedule files: {len(schedule_files)}")
    
    # Analyze by league and season
    logger.info("\n" + "=" * 60)
    logger.info("DATA BY LEAGUE AND SEASON")
    logger.info("=" * 60)
    
    leagues = ['England', 'Spain', 'Italy', 'Germany', 'France']
    seasons = ['2122', '2223', '2324']
    
    total_players = 0
    total_teams = 0
    total_matches = 0
    
    summary_data = []
    
    for league in leagues:
        logger.info(f"\n{league}:")
        for season in seasons:
            # Player stats
            player_files = [f for f in player_stats_files if f"{league}_{season}" in f]
            if player_files:
                # Use standard stats as reference
                standard_file = [f for f in player_files if 'standard' in f]
                if standard_file:
                    df = pd.read_csv(standard_file[0])
                    n_players = len(df)
                    total_players += n_players
                    
                    # Team stats
                    team_file = f"{DATA_DIR}/{league}_{season}_team_stats.csv"
                    n_teams = 0
                    if os.path.exists(team_file):
                        team_df = pd.read_csv(team_file)
                        n_teams = len(team_df)
                        total_teams += n_teams
                    
                    # Schedule
                    schedule_file = f"{DATA_DIR}/{league}_{season}_schedule.csv"
                    n_matches = 0
                    if os.path.exists(schedule_file):
                        schedule_df = pd.read_csv(schedule_file)
                        n_matches = len(schedule_df)
                        total_matches += n_matches
                    
                    logger.info(f"  {season}: {n_players} players, {n_teams} teams, {n_matches} matches")
                    
                    summary_data.append({
                        'League': league,
                        'Season': season,
                        'Players': n_players,
                        'Teams': n_teams,
                        'Matches': n_matches
                    })
    
    logger.info("\n" + "=" * 60)
    logger.info("OVERALL STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total unique player records: {total_players}")
    logger.info(f"Total team records: {total_teams}")
    logger.info(f"Total match records: {total_matches}")
    
    # Sample data inspection
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE DATA INSPECTION")
    logger.info("=" * 60)
    
    # Check one player stats file
    sample_file = player_stats_files[0]
    logger.info(f"\nInspecting: {os.path.basename(sample_file)}")
    df = pd.read_csv(sample_file)
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)[:15]}...")  # First 15 columns
    
    # Check for missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    logger.info(f"\nTop columns with missing values:")
    for col, pct in missing_pct.head(5).items():
        logger.info(f"  {col}: {pct:.1f}%")
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{DATA_DIR}/data_collection_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"\n✓ Summary saved to: {summary_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA VALIDATION COMPLETED")
    logger.info("=" * 60)
    
    return summary_df

def check_data_quality():
    """Check data quality issues"""
    logger.info("\n" + "=" * 60)
    logger.info("DATA QUALITY CHECKS")
    logger.info("=" * 60)
    
    # Check one standard stats file
    sample_file = glob.glob(f"{DATA_DIR}/*_player_stats_standard.csv")[0]
    df = pd.read_csv(sample_file)
    
    logger.info(f"\nChecking: {os.path.basename(sample_file)}")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    logger.info(f"Duplicate rows: {duplicates}")
    
    # Check data types
    logger.info(f"\nData types:")
    logger.info(df.dtypes.value_counts())
    
    # Check for negative values in key metrics
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        negative_counts = (df[numeric_cols] < 0).sum()
        if negative_counts.sum() > 0:
            logger.warning(f"\nColumns with negative values:")
            for col, count in negative_counts[negative_counts > 0].items():
                logger.warning(f"  {col}: {count} negative values")
        else:
            logger.info("\n✓ No negative values found in numeric columns")
    
    logger.info("\n" + "=" * 60)

if __name__ == "__main__":
    summary = analyze_collected_data()
    check_data_quality()
    
    logger.info("\n✓ All validation checks completed!")

