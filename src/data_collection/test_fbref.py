"""
Test script for FBref data collection
Collects data for one league and one season to verify functionality
"""

import soccerdata as sd
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test with Premier League 2023-24 season
TEST_LEAGUE = 'ENG-Premier League'
TEST_SEASON = '2324'

OUTPUT_DIR = 'data/raw/fbref'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_fbref_collection():
    """Test FBref data collection"""
    logger.info(f"Testing FBref data collection for {TEST_LEAGUE} - Season {TEST_SEASON}")
    
    try:
        fbref = sd.FBref(leagues=TEST_LEAGUE, seasons=TEST_SEASON)
        
        # Test standard stats
        logger.info("Collecting standard player stats...")
        standard_stats = fbref.read_player_season_stats(stat_type='standard')
        logger.info(f"✓ Collected {len(standard_stats)} player records")
        logger.info(f"Columns: {list(standard_stats.columns)[:10]}...")  # Show first 10 columns
        
        # Save sample
        sample_file = f"{OUTPUT_DIR}/test_sample.csv"
        standard_stats.head(20).to_csv(sample_file)
        logger.info(f"✓ Saved sample to {sample_file}")
        
        # Show sample data
        logger.info("\nSample data:")
        print(standard_stats.head())
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_fbref_collection()
    if success:
        logger.info("\n✓ Test completed successfully!")
    else:
        logger.info("\n✗ Test failed!")

