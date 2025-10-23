"""
Simple test to check if we can access Transfermarkt
Note: Full scraping may require more sophisticated approach
"""

import requests
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def test_transfermarkt_access():
    """Test if we can access Transfermarkt"""
    logger.info("Testing Transfermarkt access...")
    
    # Test URL - Premier League transfers page
    test_url = "https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/plus/?saison_id=2023&s_w=&leihe=0&intern=0"
    
    try:
        response = requests.get(test_url, headers=HEADERS, timeout=10)
        logger.info(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to find page title
            title = soup.find('title')
            if title:
                logger.info(f"Page title: {title.text.strip()}")
            
            # Check if we can find transfer tables
            tables = soup.find_all('div', class_='responsive-table')
            logger.info(f"Found {len(tables)} responsive tables")
            
            # Try to find any transfer data
            transfer_rows = soup.find_all('tr', class_=['odd', 'even'])
            logger.info(f"Found {len(transfer_rows)} potential transfer rows")
            
            if len(transfer_rows) > 0:
                logger.info("✓ Successfully accessed Transfermarkt and found transfer data")
                logger.info("\nNote: For full data collection, we may need:")
                logger.info("  - More sophisticated scraping (handling dynamic content)")
                logger.info("  - Alternative: Use pre-scraped datasets or APIs")
                logger.info("  - Alternative: Focus on FBref data and manual transfer data")
                return True
            else:
                logger.warning("⚠ Could access site but no transfer data found")
                logger.warning("  This might be due to dynamic content loading")
                return False
                
        else:
            logger.error(f"✗ Failed to access Transfermarkt (status: {response.status_code})")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error accessing Transfermarkt: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_transfermarkt_access()
    
    if not success:
        logger.info("\n" + "="*60)
        logger.info("ALTERNATIVE APPROACH:")
        logger.info("="*60)
        logger.info("Since Transfermarkt scraping may be complex, we can:")
        logger.info("1. Use existing transfer datasets (Kaggle, GitHub)")
        logger.info("2. Focus on FBref data which includes player performance")
        logger.info("3. Manually compile transfer data for key transfers")
        logger.info("4. Use transfer data from other sources")

