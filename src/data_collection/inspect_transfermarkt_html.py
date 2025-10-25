"""
Inspect Transfermarkt HTML structure to understand the page layout
"""

import requests
from bs4 import BeautifulSoup

url = "https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/saison_id/2024"

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

print(f"Fetching: {url}")
response = requests.get(url, headers=headers)

print(f"Status code: {response.status_code}")
print(f"Content length: {len(response.content)}")

# Save HTML for inspection
with open('data/external/transfermarkt_page.html', 'w', encoding='utf-8') as f:
    f.write(response.text)

print("HTML saved to: data/external/transfermarkt_page.html")

# Parse with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all divs with class 'box'
boxes = soup.find_all('div', class_='box')
print(f"\nFound {len(boxes)} 'box' divs")

# Find all tables
tables = soup.find_all('table')
print(f"Found {len(tables)} tables")

# Find tables with class 'items'
items_tables = soup.find_all('table', class_='items')
print(f"Found {len(items_tables)} 'items' tables")

# Look for responsive-table
responsive_tables = soup.find_all('div', class_='responsive-table')
print(f"Found {len(responsive_tables)} 'responsive-table' divs")

# Look for large-8 columns (common in Transfermarkt)
large_columns = soup.find_all('div', class_='large-8')
print(f"Found {len(large_columns)} 'large-8' divs")

# Print first few table headers to understand structure
if tables:
    print("\nFirst table structure:")
    first_table = tables[0]
    headers_row = first_table.find('thead')
    if headers_row:
        headers = [th.get_text(strip=True) for th in headers_row.find_all('th')]
        print(f"Headers: {headers}")
    
    # Print first few rows
    tbody = first_table.find('tbody')
    if tbody:
        rows = tbody.find_all('tr')[:3]
        print(f"First 3 rows:")
        for i, row in enumerate(rows):
            cells = [td.get_text(strip=True)[:30] for td in row.find_all('td')]
            print(f"  Row {i+1}: {cells}")

print("\n" + "="*60)
print("Check data/external/transfermarkt_page.html for full HTML")

