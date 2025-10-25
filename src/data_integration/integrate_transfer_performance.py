"""
Integrate Ewenme transfer data (2021-2022) with FBref performance data
Match players and calculate pre/post transfer performance metrics
"""

import pandas as pd
import numpy as np
import glob
import logging
from fuzzywuzzy import fuzz, process
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TransferPerformanceIntegrator:
    """Integrate transfer and performance data"""
    
    def __init__(self):
        self.fbref_dir = 'data/raw/fbref'
        self.transfer_dir = 'data/external'
        self.output_dir = 'data/processed'
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_ewenme_transfers(self) -> pd.DataFrame:
        """Load and filter Ewenme transfer data for 2021-2022"""
        logger.info("Loading Ewenme transfer data...")
        
        # Load all league transfer files
        transfer_files = {
            'Premier League': 'premier-league.csv',
            'La Liga': 'primera-division.csv',
            'Serie A': 'serie-a.csv',
            'Bundesliga': '1-bundesliga.csv',
            'Ligue 1': 'ligue-1.csv'
        }
        
        all_transfers = []
        
        for league_name, filename in transfer_files.items():
            filepath = os.path.join(self.transfer_dir, filename)
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                continue
            
            df = pd.read_csv(filepath)
            
            # Filter for 2021 and 2022 years
            df_filtered = df[df['year'].isin([2021, 2022, '2021', '2022'])].copy()
            
            logger.info(f"{league_name}: {len(df_filtered)} transfers in 2021-2022")
            
            all_transfers.append(df_filtered)
        
        # Combine all transfers
        transfers_df = pd.concat(all_transfers, ignore_index=True)
        
        logger.info(f"\nTotal transfers 2021-2022: {len(transfers_df)}")
        logger.info(f"Transfer movements: {transfers_df['transfer_movement'].value_counts().to_dict()}")
        logger.info(f"Transfers with fee: {transfers_df['fee_cleaned'].notna().sum()}")
        
        return transfers_df
    
    def load_fbref_performance(self) -> pd.DataFrame:
        """Load FBref performance data for all seasons"""
        logger.info("\nLoading FBref performance data...")
        
        # Load standard stats (contains most player info)
        standard_files = glob.glob(f"{self.fbref_dir}/*_standard.csv")
        
        all_performance = []
        
        for filepath in standard_files:
            filename = os.path.basename(filepath)
            # Extract league and season from filename
            # Format: England_2122_player_stats_standard.csv
            parts = filename.replace('_player_stats_standard.csv', '').split('_')
            
            if len(parts) < 2:
                continue
            
            # First part is country/league
            country = parts[0]
            
            # Second part is season (e.g., 2122 for 2021-22)
            season_code = parts[1]
            
            # Convert season code to readable format
            # 2122 -> 2021-22
            if len(season_code) == 4:
                year1 = season_code[:2]
                year2 = season_code[2:]
                season = f"20{year1}-{year2}"
            else:
                continue
            
            # Map country to league name
            league_map = {
                'England': 'Premier League',
                'Spain': 'La Liga',
                'Italy': 'Serie A',
                'Germany': 'Bundesliga',
                'France': 'Ligue 1'
            }
            league = league_map.get(country, country)
            
            # FBref CSVs have multi-level headers
            df = pd.read_csv(filepath, header=[0,1], skiprows=[2])
            
            # Flatten column names
            df.columns = ['_'.join(col).strip() if col[1] and 'Unnamed' not in col[1] else col[0] for col in df.columns.values]
            
            # Rename first 4 columns
            col_list = list(df.columns)
            col_list[0] = 'league_orig'
            col_list[1] = 'season_orig'
            col_list[2] = 'team'
            col_list[3] = 'player'
            df.columns = col_list
            
            df['league'] = league
            df['season'] = season
            
            all_performance.append(df)
        
        performance_df = pd.concat(all_performance, ignore_index=True)
        
        logger.info(f"Total performance records: {len(performance_df)}")
        logger.info(f"Unique players: {performance_df['player'].nunique()}")
        logger.info(f"Seasons: {sorted(performance_df['season'].unique())}")
        logger.info(f"Leagues: {performance_df['league'].unique()}")
        
        return performance_df
    
    def normalize_player_name(self, name: str) -> str:
        """Normalize player name for matching"""
        if pd.isna(name):
            return ""
        
        # Convert to lowercase
        name = str(name).lower().strip()
        
        # Remove accents and special characters (comprehensive)
        import unicodedata
        name = unicodedata.normalize('NFKD', name)
        name = ''.join([c for c in name if not unicodedata.combining(c)])
        
        # Remove extra spaces
        name = ' '.join(name.split())
        
        return name
    
    def match_player_fuzzy(self, transfer_name: str, performance_df: pd.DataFrame, 
                          threshold: int = 75) -> pd.DataFrame:
        """
        Match a transfer player name with performance data using fuzzy matching
        
        Args:
            transfer_name: Player name from transfer data
            performance_df: DataFrame with performance data
            threshold: Minimum similarity score (0-100)
        
        Returns:
            Matched performance records or empty DataFrame
        """
        if pd.isna(transfer_name):
            return pd.DataFrame()
        
        # Normalize transfer name
        norm_transfer_name = self.normalize_player_name(transfer_name)
        
        # Get unique player names from performance data
        perf_names = performance_df['player'].unique()
        
        # Find best match using fuzzy matching
        best_match = process.extractOne(
            norm_transfer_name,
            [self.normalize_player_name(name) for name in perf_names],
            scorer=fuzz.ratio
        )
        
        if best_match and best_match[1] >= threshold:
            # Get the original name (not normalized)
            matched_idx = [self.normalize_player_name(name) for name in perf_names].index(best_match[0])
            matched_name = perf_names[matched_idx]
            
            # Return all records for this player
            return performance_df[performance_df['player'] == matched_name].copy()
        
        return pd.DataFrame()
    
    def integrate_data(self) -> pd.DataFrame:
        """
        Main integration function
        Match transfers with performance data and calculate metrics
        """
        logger.info("="*60)
        logger.info("STARTING DATA INTEGRATION")
        logger.info("="*60)
        
        # Load data
        transfers_df = self.load_ewenme_transfers()
        performance_df = self.load_fbref_performance()
        
        # Focus on arrivals (transfers IN) for now
        arrivals_df = transfers_df[transfers_df['transfer_movement'] == 'in'].copy()
        logger.info(f"\nFocusing on {len(arrivals_df)} arrivals (transfers IN)")
        
        # Match players
        logger.info("\nMatching players with performance data...")
        
        matched_transfers = []
        match_count = 0
        no_match_count = 0
        
        for idx, transfer in arrivals_df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Progress: {idx}/{len(arrivals_df)} ({idx/len(arrivals_df)*100:.1f}%)")
            
            player_name = transfer['player_name']
            transfer_year = transfer['year']
            club_name = transfer['club_name']
            
            # Match player in performance data
            matched_perf = self.match_player_fuzzy(player_name, performance_df)
            
            if matched_perf.empty:
                no_match_count += 1
                continue
            
            # Get performance before and after transfer
            # Transfer year 2021 means 2021/22 season
            # We want performance from previous season (2020/21) and current season (2021/22)
            
            # Convert transfer_year to int if needed
            transfer_year_int = int(transfer_year)
            # Use hyphen format to match FBref: 2021-22
            transfer_season = f"{transfer_year_int}-{str(transfer_year_int+1)[-2:]}"
            prev_year = transfer_year_int - 1
            prev_season = f"{prev_year}-{str(transfer_year_int)[-2:]}"
            
            # Performance before transfer (previous season)
            perf_before = matched_perf[matched_perf['season'] == prev_season]
            
            # Performance after transfer (current season at new club)
            perf_after = matched_perf[matched_perf['season'] == transfer_season]
            
            if perf_before.empty and perf_after.empty:
                no_match_count += 1
                continue
            
            # Combine transfer and performance data
            transfer_record = transfer.to_dict()
            
            # Add performance metrics
            if not perf_before.empty:
                perf_before_agg = perf_before.iloc[0]  # Take first record if multiple
                transfer_record['perf_before_goals'] = perf_before_agg.get('Performance_Gls', np.nan)
                transfer_record['perf_before_assists'] = perf_before_agg.get('Performance_Ast', np.nan)
                transfer_record['perf_before_minutes'] = perf_before_agg.get('Playing Time_Min', np.nan)
                transfer_record['perf_before_matches'] = perf_before_agg.get('Playing Time_MP', np.nan)
            
            if not perf_after.empty:
                perf_after_agg = perf_after.iloc[0]
                transfer_record['perf_after_goals'] = perf_after_agg.get('Performance_Gls', np.nan)
                transfer_record['perf_after_assists'] = perf_after_agg.get('Performance_Ast', np.nan)
                transfer_record['perf_after_minutes'] = perf_after_agg.get('Playing Time_Min', np.nan)
                transfer_record['perf_after_matches'] = perf_after_agg.get('Playing Time_MP', np.nan)
            
            matched_transfers.append(transfer_record)
            match_count += 1
        
        logger.info(f"\nMatching complete:")
        logger.info(f"  Matched: {match_count}")
        logger.info(f"  No match: {no_match_count}")
        logger.info(f"  Match rate: {match_count/(match_count+no_match_count)*100:.1f}%")
        
        # Create final DataFrame
        integrated_df = pd.DataFrame(matched_transfers)
        
        # Save integrated data
        output_file = os.path.join(self.output_dir, 'integrated_transfers_performance.csv')
        integrated_df.to_csv(output_file, index=False)
        logger.info(f"\n✅ Integrated data saved to: {output_file}")
        
        return integrated_df


def main():
    """Main execution"""
    integrator = TransferPerformanceIntegrator()
    integrated_df = integrator.integrate_data()
    
    # Display summary
    logger.info("\n" + "="*60)
    logger.info("INTEGRATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total integrated records: {len(integrated_df)}")
    logger.info(f"\nColumns: {list(integrated_df.columns)}")
    logger.info(f"\nSample data:")
    cols_to_show = ['player_name', 'age', 'club_name', 'fee_cleaned', 
                    'perf_before_goals', 'perf_after_goals']
    logger.info(integrated_df[cols_to_show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

