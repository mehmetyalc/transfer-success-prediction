"""Test script for V2 scraper"""

import sys
sys.path.append('src/data_collection')

from scrape_transfermarkt_v2 import TransfermarktScraperV2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Testing Transfermarkt V2 scraper with Premier League 2024/25...")
    
    scraper = TransfermarktScraperV2()
    
    # Test with Premier League 2024/25
    df = scraper.get_league_transfers('GB1', 2024)
    
    if df.empty:
        logger.error("❌ No data scraped!")
        return
    
    logger.info(f"\n✅ Successfully scraped {len(df)} transfers!")
    
    # Display sample
    logger.info(f"\nSample data (first 10 rows):")
    cols_to_show = ['player_name', 'age', 'position', 'club_name', 'transfer_movement', 'fee', 'market_value']
    logger.info(df[cols_to_show].head(10).to_string(index=False))
    
    # Statistics
    logger.info(f"\n{'='*60}")
    logger.info("STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"Total transfers: {len(df)}")
    logger.info(f"Arrivals: {(df['transfer_movement'] == 'in').sum()}")
    logger.info(f"Departures: {(df['transfer_movement'] == 'out').sum()}")
    logger.info(f"Transfers with fee: {df['fee_cleaned'].notna().sum()} ({df['fee_cleaned'].notna().sum()/len(df)*100:.1f}%)")
    logger.info(f"Free transfers: {(df['fee_cleaned'] == 0).sum()}")
    
    if df['fee_cleaned'].notna().sum() > 0:
        logger.info(f"\nFee statistics:")
        logger.info(f"  Mean: €{df['fee_cleaned'].mean():.2f}M")
        logger.info(f"  Median: €{df['fee_cleaned'].median():.2f}M")
        logger.info(f"  Max: €{df['fee_cleaned'].max():.2f}M")
    
    # Save test output
    df.to_csv('data/external/test_v2_premier_league_2024_25.csv', index=False)
    logger.info(f"\n✅ Test data saved to: data/external/test_v2_premier_league_2024_25.csv")

if __name__ == "__main__":
    main()

