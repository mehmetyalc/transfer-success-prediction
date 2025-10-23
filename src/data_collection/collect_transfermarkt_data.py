"""
Transfermarkt Data Collection Script
Collects transfer data including fees, market values, and player information
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import logging
from datetime import datetime
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = 'data/raw/transfermarkt'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Headers for requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Base URL
BASE_URL = 'https://www.transfermarkt.com'

# League URLs (English versions)
LEAGUE_URLS = {
    'Premier League': '/premier-league/transfers/wettbewerb/GB1',
    'La Liga': '/laliga/transfers/wettbewerb/ES1',
    'Serie A': '/serie-a/transfers/wettbewerb/IT1',
    'Bundesliga': '/bundesliga/transfers/wettbewerb/L1',
    'Ligue 1': '/ligue-1/transfers/wettbewerb/FR1'
}

# Seasons to collect
SEASONS = ['2021', '2022', '2023']


def get_page(url, retries=3, delay=2):
    """
    Get page content with retry logic
    
    Args:
        url (str): URL to fetch
        retries (int): Number of retries
        delay (int): Delay between retries in seconds
    
    Returns:
        BeautifulSoup: Parsed HTML content
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            time.sleep(delay)  # Be respectful to the server
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay * 2)
            else:
                logger.error(f"Failed to fetch {url} after {retries} attempts")
                return None


def clean_transfer_fee(fee_str):
    """
    Clean and convert transfer fee string to numeric value
    
    Args:
        fee_str (str): Transfer fee string (e.g., '€50.00m', 'Free transfer')
    
    Returns:
        float: Transfer fee in millions of euros
    """
    if not fee_str or fee_str == '-' or 'Free' in fee_str or 'Loan' in fee_str:
        return 0.0
    
    # Remove currency symbol and spaces
    fee_str = fee_str.replace('€', '').replace(' ', '').strip()
    
    # Convert to millions
    if 'm' in fee_str:
        return float(fee_str.replace('m', ''))
    elif 'k' in fee_str or 'Th.' in fee_str:
        return float(fee_str.replace('k', '').replace('Th.', '')) / 1000
    else:
        try:
            return float(fee_str)
        except:
            return 0.0


def scrape_league_transfers(league_name, league_url, season):
    """
    Scrape transfer data for a specific league and season
    
    Args:
        league_name (str): Name of the league
        league_url (str): URL path for the league
        season (str): Season year (e.g., '2021' for 2021/22)
    
    Returns:
        pd.DataFrame: Transfer data
    """
    logger.info(f"Scraping {league_name} transfers for {season}/{int(season)+1} season")
    
    # Construct URL with season parameter
    url = f"{BASE_URL}{league_url}/plus/?saison_id={season}&s_w=&leihe=0&intern=0"
    
    soup = get_page(url)
    if not soup:
        return None
    
    transfers = []
    
    try:
        # Find transfer tables (arrivals and departures)
        tables = soup.find_all('div', class_='responsive-table')
        
        for table in tables:
            rows = table.find_all('tr', class_=['odd', 'even'])
            
            for row in rows:
                try:
                    # Extract player information
                    player_cell = row.find('td', class_='hauptlink')
                    if not player_cell:
                        continue
                    
                    player_link = player_cell.find('a')
                    if not player_link:
                        continue
                    
                    player_name = player_link.text.strip()
                    player_url = player_link.get('href', '')
                    
                    # Extract position
                    position_cell = row.find_all('td')[1] if len(row.find_all('td')) > 1 else None
                    position = position_cell.text.strip() if position_cell else 'Unknown'
                    
                    # Extract age
                    age_cell = row.find('td', class_='zentriert')
                    age = age_cell.text.strip() if age_cell else 'Unknown'
                    
                    # Extract clubs
                    club_cells = row.find_all('td', class_='no-border-links')
                    from_club = club_cells[0].text.strip() if len(club_cells) > 0 else 'Unknown'
                    to_club = club_cells[1].text.strip() if len(club_cells) > 1 else 'Unknown'
                    
                    # Extract market value
                    market_value_cell = row.find('td', class_='rechts')
                    market_value = market_value_cell.text.strip() if market_value_cell else '€0'
                    
                    # Extract transfer fee
                    fee_cell = row.find('td', class_='rechts hauptlink')
                    transfer_fee = fee_cell.text.strip() if fee_cell else '€0'
                    
                    transfer = {
                        'player_name': player_name,
                        'player_url': player_url,
                        'position': position,
                        'age': age,
                        'from_club': from_club,
                        'to_club': to_club,
                        'league': league_name,
                        'season': f"{season}/{int(season)+1}",
                        'transfer_fee_raw': transfer_fee,
                        'transfer_fee_millions': clean_transfer_fee(transfer_fee),
                        'market_value_raw': market_value,
                        'market_value_millions': clean_transfer_fee(market_value)
                    }
                    
                    transfers.append(transfer)
                    
                except Exception as e:
                    logger.warning(f"Error parsing row: {str(e)}")
                    continue
        
        logger.info(f"  ✓ Collected {len(transfers)} transfers for {league_name} - {season}")
        return pd.DataFrame(transfers)
        
    except Exception as e:
        logger.error(f"  ✗ Error scraping {league_name} - {season}: {str(e)}")
        return None


def collect_all_transfers():
    """
    Collect transfer data for all leagues and seasons
    """
    logger.info("=" * 60)
    logger.info("Starting Transfermarkt Data Collection")
    logger.info("=" * 60)
    logger.info(f"Leagues: {list(LEAGUE_URLS.keys())}")
    logger.info(f"Seasons: {SEASONS}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 60)
    
    all_transfers = []
    
    for league_name, league_url in LEAGUE_URLS.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {league_name}")
        logger.info(f"{'=' * 60}")
        
        for season in SEASONS:
            transfers_df = scrape_league_transfers(league_name, league_url, season)
            
            if transfers_df is not None and not transfers_df.empty:
                # Save individual league-season file
                filename = f"{OUTPUT_DIR}/{league_name.replace(' ', '_')}_{season}_transfers.csv"
                transfers_df.to_csv(filename, index=False)
                logger.info(f"  → Saved: {filename}")
                
                all_transfers.append(transfers_df)
            
            # Be respectful to the server
            time.sleep(3)
    
    # Combine all transfers into one file
    if all_transfers:
        combined_df = pd.concat(all_transfers, ignore_index=True)
        combined_filename = f"{OUTPUT_DIR}/all_transfers_combined.csv"
        combined_df.to_csv(combined_filename, index=False)
        logger.info(f"\n✓ Saved combined file: {combined_filename}")
        logger.info(f"Total transfers collected: {len(combined_df)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Transfermarkt data collection completed")
    logger.info("=" * 60)


def main():
    """
    Main function
    """
    try:
        collect_all_transfers()
    except KeyboardInterrupt:
        logger.info("\nData collection interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()

