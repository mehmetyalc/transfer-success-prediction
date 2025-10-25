"""
Transfermarkt Web Scraper for 2023-2024 Seasons
Collects transfer data for Premier League, La Liga, Serie A, Bundesliga, and Ligue 1
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
from typing import List, Dict
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TransfermarktScraper:
    """Scraper for Transfermarkt transfer data"""
    
    def __init__(self):
        self.base_url = "https://www.transfermarkt.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive',
        }
        
        self.leagues = {
            'GB1': {'name': 'Premier League', 'slug': 'premier-league'},
            'ES1': {'name': 'La Liga', 'slug': 'laliga'},
            'IT1': {'name': 'Serie A', 'slug': 'serie-a'},
            'L1': {'name': 'Bundesliga', 'slug': '1-bundesliga'},
            'FR1': {'name': 'Ligue 1', 'slug': 'ligue-1'}
        }
    
    def clean_fee(self, fee_text: str) -> float:
        """
        Convert fee text to numeric value in millions EUR
        
        Examples:
            '€116.60m' -> 116.60
            '€40.00m' -> 40.00
            'free transfer' -> 0.0
            'loan' -> None
            '?' -> None
        """
        if not fee_text or fee_text == '-':
            return None
        
        fee_text = fee_text.strip().lower()
        
        # Free transfer
        if 'free' in fee_text or 'end of loan' in fee_text:
            return 0.0
        
        # Loan
        if 'loan' in fee_text:
            return None
        
        # Unknown
        if '?' in fee_text or fee_text == '-':
            return None
        
        # Extract numeric value
        # Remove currency symbol and 'm' suffix
        fee_text = fee_text.replace('€', '').replace('m', '').replace(',', '.')
        
        try:
            # Check for 'k' (thousands)
            if 'k' in fee_text:
                value = float(fee_text.replace('k', ''))
                return value / 1000  # Convert to millions
            else:
                return float(fee_text)
        except ValueError:
            logger.warning(f"Could not parse fee: {fee_text}")
            return None
    
    def clean_market_value(self, mv_text: str) -> float:
        """Convert market value text to numeric value in millions EUR"""
        return self.clean_fee(mv_text)
    
    def get_league_transfers(self, league_id: str, season: int, transfer_window: str = '') -> pd.DataFrame:
        """
        Scrape transfers for a specific league and season
        
        Args:
            league_id: League code (GB1, ES1, IT1, L1, FR1)
            season: Season year (2023 for 2023/24)
            transfer_window: 's' for summer, 'w' for winter, '' for both
        
        Returns:
            DataFrame with transfer data
        """
        league_info = self.leagues.get(league_id, {'name': league_id, 'slug': league_id})
        league_name = league_info['name']
        league_slug = league_info['slug']
        season_str = f"{season}/{str(season+1)[-2:]}"
        
        logger.info(f"Scraping {league_name} - {season_str} transfers...")
        
        # Construct URL
        # Format: /premier-league/transfers/wettbewerb/GB1/saison_id/2023
        url = f"{self.base_url}/{league_slug}/transfers/wettbewerb/{league_id}/saison_id/{season}"
        
        # Add filters if needed
        if transfer_window:
            url += f"/s_w/{transfer_window}"
        
        # Rate limiting
        time.sleep(random.uniform(3, 6))
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return pd.DataFrame()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        transfers = []
        
        # Find all club sections
        club_boxes = soup.find_all('div', class_='box')
        
        for club_box in club_boxes:
            # Get club name from header
            club_header = club_box.find('h2')
            if not club_header:
                continue
            
            club_name_elem = club_header.find('a')
            if not club_name_elem:
                continue
            
            club_name = club_name_elem.get_text(strip=True)
            
            # Find transfer tables (arrivals and departures)
            tables = club_box.find_all('table', class_='items')
            
            for table in tables:
                # Determine if this is arrivals or departures
                table_header = table.find_previous('div', class_='table-header')
                if not table_header:
                    continue
                
                header_text = table_header.get_text(strip=True).lower()
                is_arrival = 'in' in header_text or 'arrivals' in header_text
                
                # Parse rows
                rows = table.find('tbody').find_all('tr') if table.find('tbody') else []
                
                for row in rows:
                    # Skip header rows
                    if row.find('th'):
                        continue
                    
                    cells = row.find_all('td')
                    if len(cells) < 7:
                        continue
                    
                    try:
                        # Extract player info
                        player_cell = cells[0]
                        player_link = player_cell.find('a', class_='spielprofil_tooltip')
                        if not player_link:
                            continue
                        
                        player_name = player_link.get_text(strip=True)
                        
                        # Age
                        age_text = cells[1].get_text(strip=True)
                        try:
                            age = int(age_text) if age_text else None
                        except ValueError:
                            age = None
                        
                        # Position
                        position = cells[2].get_text(strip=True)
                        
                        # Market value
                        mv_cell = cells[3]
                        mv_text = mv_cell.get_text(strip=True)
                        market_value = self.clean_market_value(mv_text)
                        
                        # Other club
                        other_club_cell = cells[4]
                        other_club_link = other_club_cell.find('a')
                        other_club = other_club_link.get_text(strip=True) if other_club_link else None
                        
                        # Fee
                        fee_cell = cells[5]
                        fee_text = fee_cell.get_text(strip=True)
                        fee = self.clean_fee(fee_text)
                        
                        # Construct transfer record
                        transfer = {
                            'player_name': player_name,
                            'age': age,
                            'position': position,
                            'market_value': market_value,
                            'fee_cleaned': fee,
                            'fee': fee_text,
                            'league_name': league_name,
                            'league_id': league_id,
                            'season': season_str,
                            'year': str(season),
                            'transfer_period': transfer_window if transfer_window else 'both',
                        }
                        
                        if is_arrival:
                            transfer['club_name'] = club_name
                            transfer['club_involved_name'] = other_club
                            transfer['transfer_movement'] = 'in'
                        else:
                            transfer['club_name'] = other_club
                            transfer['club_involved_name'] = club_name
                            transfer['transfer_movement'] = 'out'
                        
                        transfers.append(transfer)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing row: {e}")
                        continue
        
        df = pd.DataFrame(transfers)
        logger.info(f"Scraped {len(df)} transfers for {league_name} {season_str}")
        
        return df
    
    def scrape_all_leagues_seasons(self, seasons: List[int]) -> pd.DataFrame:
        """
        Scrape all leagues for multiple seasons
        
        Args:
            seasons: List of season years (e.g., [2023, 2024])
        
        Returns:
            Combined DataFrame with all transfers
        """
        all_transfers = []
        
        total_combinations = len(self.leagues) * len(seasons)
        current = 0
        
        for league_id, league_name in self.leagues.items():
            for season in seasons:
                current += 1
                logger.info(f"Progress: {current}/{total_combinations}")
                
                df = self.get_league_transfers(league_id, season)
                
                if not df.empty:
                    all_transfers.append(df)
                
                # Rate limiting between requests
                time.sleep(random.uniform(5, 10))
        
        if not all_transfers:
            logger.error("No transfers scraped!")
            return pd.DataFrame()
        
        # Combine all dataframes
        final_df = pd.concat(all_transfers, ignore_index=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SCRAPING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total transfers scraped: {len(final_df)}")
        logger.info(f"Leagues: {final_df['league_name'].nunique()}")
        logger.info(f"Seasons: {final_df['season'].nunique()}")
        logger.info(f"\nBreakdown by league:")
        logger.info(final_df['league_name'].value_counts().to_string())
        
        return final_df


def main():
    """Main execution function"""
    logger.info("="*60)
    logger.info("TRANSFERMARKT SCRAPER - 2023/24 and 2024/25 SEASONS")
    logger.info("="*60)
    
    scraper = TransfermarktScraper()
    
    # Scrape 2023 and 2024 seasons
    seasons = [2023, 2024]
    
    df = scraper.scrape_all_leagues_seasons(seasons)
    
    if df.empty:
        logger.error("No data scraped!")
        return
    
    # Save to CSV
    output_file = 'data/external/transfermarkt_2023_2024.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"\n✅ Data saved to: {output_file}")
    
    # Display sample
    logger.info(f"\nSample data (first 5 rows):")
    logger.info(df.head().to_string())
    
    # Statistics
    logger.info(f"\n{'='*60}")
    logger.info("STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"Total transfers: {len(df)}")
    logger.info(f"Transfers with fee data: {df['fee_cleaned'].notna().sum()} ({df['fee_cleaned'].notna().sum()/len(df)*100:.1f}%)")
    logger.info(f"Free transfers: {(df['fee_cleaned'] == 0).sum()}")
    logger.info(f"Transfers with market value: {df['market_value'].notna().sum()} ({df['market_value'].notna().sum()/len(df)*100:.1f}%)")
    
    if df['fee_cleaned'].notna().sum() > 0:
        logger.info(f"\nFee statistics (EUR millions):")
        logger.info(f"  Mean: €{df['fee_cleaned'].mean():.2f}M")
        logger.info(f"  Median: €{df['fee_cleaned'].median():.2f}M")
        logger.info(f"  Max: €{df['fee_cleaned'].max():.2f}M")
        logger.info(f"  Total: €{df['fee_cleaned'].sum():.2f}M")


if __name__ == "__main__":
    main()

