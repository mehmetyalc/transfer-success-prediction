"""
Analyze downloaded transfer data from ewenme/transfers repository
Check data quality, coverage, and suitability for our project
"""

import pandas as pd
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = 'data/external'

def analyze_transfer_files():
    """Analyze all downloaded transfer CSV files"""
    logger.info("=" * 60)
    logger.info("TRANSFER DATA ANALYSIS")
    logger.info("=" * 60)
    
    # Get all CSV files
    csv_files = glob.glob(f"{DATA_DIR}/*.csv")
    
    if not csv_files:
        logger.error("No CSV files found in data/external/")
        return
    
    logger.info(f"\nFound {len(csv_files)} transfer data files:")
    for f in csv_files:
        logger.info(f"  - {os.path.basename(f)}")
    
    # Analyze each file
    all_data = []
    
    for file_path in csv_files:
        league_name = os.path.basename(file_path).replace('.csv', '')
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing: {league_name}")
        logger.info(f"{'='*60}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Basic info
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Check years covered
            if 'year' in df.columns:
                years = df['year'].unique()
                years_sorted = sorted([int(y) for y in years if pd.notna(y)])
                logger.info(f"Years covered: {min(years_sorted)} to {max(years_sorted)}")
                
                # Focus on our target years (2021-2024)
                target_years = [2021, 2022, 2023, 2024]
                df_target = df[df['year'].isin([str(y) for y in target_years])]
                logger.info(f"Records in 2021-2024: {len(df_target)} ({len(df_target)/len(df)*100:.1f}%)")
                
                # Year breakdown
                logger.info("\nYear breakdown (2021-2024):")
                for year in target_years:
                    year_count = len(df[df['year'] == str(year)])
                    if year_count > 0:
                        logger.info(f"  {year}: {year_count} transfers")
            
            # Check transfer movement
            if 'transfer_movement' in df.columns:
                logger.info(f"\nTransfer movements:")
                movement_counts = df['transfer_movement'].value_counts()
                for movement, count in movement_counts.items():
                    logger.info(f"  {movement}: {count}")
            
            # Check fee data
            if 'fee_cleaned' in df.columns:
                fee_data = df['fee_cleaned'].dropna()
                logger.info(f"\nTransfer fees:")
                logger.info(f"  Records with fee: {len(fee_data)} ({len(fee_data)/len(df)*100:.1f}%)")
                if len(fee_data) > 0:
                    logger.info(f"  Mean fee: €{fee_data.mean():.2f}M")
                    logger.info(f"  Median fee: €{fee_data.median():.2f}M")
                    logger.info(f"  Max fee: €{fee_data.max():.2f}M")
            
            # Check missing data
            logger.info(f"\nMissing data:")
            missing = df.isnull().sum()
            for col, count in missing[missing > 0].items():
                logger.info(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
            
            # Sample data
            logger.info(f"\nSample records (first 3):")
            sample_cols = ['player_name', 'club_name', 'year', 'transfer_movement', 'fee_cleaned']
            available_cols = [col for col in sample_cols if col in df.columns]
            logger.info(df[available_cols].head(3).to_string(index=False))
            
            # Store for summary
            all_data.append({
                'league': league_name,
                'total_records': len(df),
                'records_2021_2024': len(df[df['year'].isin(['2021', '2022', '2023', '2024'])]) if 'year' in df.columns else 0,
                'has_fee_data': 'fee_cleaned' in df.columns,
                'fee_coverage': len(df['fee_cleaned'].dropna()) / len(df) * 100 if 'fee_cleaned' in df.columns else 0
            })
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    summary_df = pd.DataFrame(all_data)
    logger.info(f"\n{summary_df.to_string(index=False)}")
    
    total_records = summary_df['total_records'].sum()
    total_2021_2024 = summary_df['records_2021_2024'].sum()
    
    logger.info(f"\nTotal records across all leagues: {total_records}")
    logger.info(f"Total records in 2021-2024: {total_2021_2024}")
    logger.info(f"Average fee coverage: {summary_df['fee_coverage'].mean():.1f}%")
    
    logger.info("\n" + "=" * 60)
    logger.info("SUITABILITY FOR PROJECT")
    logger.info("=" * 60)
    
    if total_2021_2024 > 0:
        logger.info("✓ Data covers our target period (2021-2024)")
    else:
        logger.warning("⚠ Limited data for target period (2021-2024)")
    
    if summary_df['fee_coverage'].mean() > 50:
        logger.info(f"✓ Good fee coverage ({summary_df['fee_coverage'].mean():.1f}%)")
    else:
        logger.warning(f"⚠ Limited fee coverage ({summary_df['fee_coverage'].mean():.1f}%)")
    
    logger.info("\nNext steps:")
    logger.info("1. Merge transfer data with FBref performance data")
    logger.info("2. Match players by name (fuzzy matching may be needed)")
    logger.info("3. Calculate pre-transfer and post-transfer performance metrics")
    logger.info("4. Build features for modeling")

if __name__ == "__main__":
    analyze_transfer_files()

