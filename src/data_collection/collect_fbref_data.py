"""
FBref Data Collection Script
Collects player and team statistics from FBref using soccerdata library
"""

import soccerdata as sd
import pandas as pd
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define leagues and seasons to collect
LEAGUES = {
    'ENG-Premier League': 'England',
    'ESP-La Liga': 'Spain',
    'ITA-Serie A': 'Italy',
    'GER-Bundesliga': 'Germany',
    'FRA-Ligue 1': 'France'
}

# Collect data from last 3 seasons (adjust years as needed)
SEASONS = ['2122', '2223', '2324']  # 2021-22, 2022-23, 2023-24

# Output directory
OUTPUT_DIR = 'data/raw/fbref'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def collect_player_stats(league, season):
    """
    Collect player statistics for a given league and season
    
    Args:
        league (str): League identifier (e.g., 'ENG-Premier League')
        season (str): Season identifier (e.g., '2122' for 2021-22)
    
    Returns:
        dict: Dictionary containing different stat types
    """
    logger.info(f"Collecting player stats for {league} - Season {season}")
    
    try:
        fbref = sd.FBref(leagues=league, seasons=season)
        
        stats_data = {}
        
        # Standard stats (goals, assists, minutes, etc.)
        logger.info("  - Collecting standard stats...")
        stats_data['standard'] = fbref.read_player_season_stats(stat_type='standard')
        
        # Shooting stats (goals, shots, xG)
        logger.info("  - Collecting shooting stats...")
        stats_data['shooting'] = fbref.read_player_season_stats(stat_type='shooting')
        
        # Passing stats (assists, xA, pass completion)
        logger.info("  - Collecting passing stats...")
        stats_data['passing'] = fbref.read_player_season_stats(stat_type='passing')
        
        # Playing time
        logger.info("  - Collecting playing time stats...")
        stats_data['playing_time'] = fbref.read_player_season_stats(stat_type='playing_time')
        
        logger.info(f"  ✓ Successfully collected stats for {league} - {season}")
        return stats_data
        
    except Exception as e:
        logger.error(f"  ✗ Error collecting stats for {league} - {season}: {str(e)}")
        return None


def collect_team_stats(league, season):
    """
    Collect team statistics for a given league and season
    
    Args:
        league (str): League identifier
        season (str): Season identifier
    
    Returns:
        pd.DataFrame: Team statistics
    """
    logger.info(f"Collecting team stats for {league} - Season {season}")
    
    try:
        fbref = sd.FBref(leagues=league, seasons=season)
        
        # Team season stats
        team_stats = fbref.read_team_season_stats(stat_type='standard')
        
        logger.info(f"  ✓ Successfully collected team stats for {league} - {season}")
        return team_stats
        
    except Exception as e:
        logger.error(f"  ✗ Error collecting team stats for {league} - {season}: {str(e)}")
        return None


def collect_schedule(league, season):
    """
    Collect match schedule/results for a given league and season
    
    Args:
        league (str): League identifier
        season (str): Season identifier
    
    Returns:
        pd.DataFrame: Match schedule and results
    """
    logger.info(f"Collecting schedule for {league} - Season {season}")
    
    try:
        fbref = sd.FBref(leagues=league, seasons=season)
        schedule = fbref.read_schedule()
        
        logger.info(f"  ✓ Successfully collected schedule for {league} - {season}")
        return schedule
        
    except Exception as e:
        logger.error(f"  ✗ Error collecting schedule for {league} - {season}: {str(e)}")
        return None


def save_data(data, league, season, data_type):
    """
    Save collected data to CSV files
    
    Args:
        data: Data to save (DataFrame or dict of DataFrames)
        league (str): League identifier
        season (str): Season identifier
        data_type (str): Type of data (player_stats, team_stats, schedule)
    """
    league_name = LEAGUES.get(league, league).replace(' ', '_')
    
    if isinstance(data, dict):
        # Multiple stat types (player stats)
        for stat_type, df in data.items():
            if df is not None and not df.empty:
                filename = f"{OUTPUT_DIR}/{league_name}_{season}_{data_type}_{stat_type}.csv"
                df.to_csv(filename)
                logger.info(f"  → Saved: {filename} ({len(df)} records)")
    else:
        # Single DataFrame
        if data is not None and not data.empty:
            filename = f"{OUTPUT_DIR}/{league_name}_{season}_{data_type}.csv"
            data.to_csv(filename)
            logger.info(f"  → Saved: {filename} ({len(data)} records)")


def main():
    """
    Main function to orchestrate data collection
    """
    logger.info("=" * 60)
    logger.info("Starting FBref Data Collection")
    logger.info("=" * 60)
    logger.info(f"Leagues: {list(LEAGUES.keys())}")
    logger.info(f"Seasons: {SEASONS}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 60)
    
    total_start = datetime.now()
    
    for league in LEAGUES.keys():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {league}")
        logger.info(f"{'=' * 60}")
        
        for season in SEASONS:
            logger.info(f"\nSeason: {season}")
            logger.info("-" * 40)
            
            # Collect player stats
            player_stats = collect_player_stats(league, season)
            if player_stats is not None:
                save_data(player_stats, league, season, 'player_stats')
            
            # Collect team stats
            team_stats = collect_team_stats(league, season)
            if team_stats is not None:
                save_data(team_stats, league, season, 'team_stats')
            
            # Collect schedule
            schedule = collect_schedule(league, season)
            if schedule is not None:
                save_data(schedule, league, season, 'schedule')
            
            logger.info("-" * 40)
    
    total_time = datetime.now() - total_start
    logger.info("\n" + "=" * 60)
    logger.info(f"Data collection completed in {total_time}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

