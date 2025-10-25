"""
Advanced Feature Engineering for Transfer Success Prediction
Create features from existing data and enrich with additional metrics
"""

import pandas as pd
import numpy as np
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TransferFeatureEngineer:
    """Create advanced features for transfer success prediction"""
    
    def __init__(self):
        self.data_dir = 'data/processed'
        self.fbref_dir = 'data/raw/fbref'
        self.output_dir = 'data/processed'
    
    def load_base_data(self) -> pd.DataFrame:
        """Load the processed transfer data with basic features"""
        logger.info("Loading base transfer data...")
        df = pd.read_csv(f'{self.data_dir}/transfers_with_features.csv')
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def load_fbref_extended(self) -> pd.DataFrame:
        """Load FBref data with all available metrics"""
        logger.info("Loading extended FBref performance data...")
        
        all_data = []
        
        # Load all standard stats files
        standard_files = glob.glob(f"{self.fbref_dir}/*_standard.csv")
        
        for filepath in standard_files:
            # Parse filename
            import os
            filename = os.path.basename(filepath)
            parts = filename.replace('_player_stats_standard.csv', '').split('_')
            
            if len(parts) < 2:
                continue
            
            country = parts[0]
            season_code = parts[1]
            
            # Convert season code
            if len(season_code) == 4:
                year1 = season_code[:2]
                year2 = season_code[2:]
                season = f"20{year1}-{year2}"
            else:
                continue
            
            # Map country to league
            league_map = {
                'England': 'Premier League',
                'Spain': 'La Liga',
                'Italy': 'Serie A',
                'Germany': 'Bundesliga',
                'France': 'Ligue 1'
            }
            league = league_map.get(country, country)
            
            # Load with multi-level headers
            df = pd.read_csv(filepath, header=[0,1], skiprows=[2])
            
            # Flatten column names
            df.columns = ['_'.join(col).strip() if col[1] and 'Unnamed' not in col[1] else col[0] 
                         for col in df.columns.values]
            
            # Rename first 4 columns
            col_list = list(df.columns)
            col_list[0] = 'league_orig'
            col_list[1] = 'season_orig'
            col_list[2] = 'team'
            col_list[3] = 'player'
            df.columns = col_list
            
            df['league'] = league
            df['season'] = season
            
            all_data.append(df)
        
        perf_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(perf_df)} performance records")
        
        return perf_df
    
    def create_performance_trend_features(self, df: pd.DataFrame, perf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on performance trends before transfer
        """
        logger.info("Creating performance trend features...")
        
        # For each transfer, get performance from 2 seasons before if available
        # This would require 2020-21 data which we don't have
        # So we'll use available data to create trend indicators
        
        # Performance efficiency metrics
        df['goals_per_90_before'] = (df['perf_before_goals'] / (df['perf_before_minutes'] / 90)).replace([np.inf, -np.inf], 0).fillna(0)
        df['goals_per_90_after'] = (df['perf_after_goals'] / (df['perf_after_minutes'] / 90)).replace([np.inf, -np.inf], 0).fillna(0)
        
        df['assists_per_90_before'] = (df['perf_before_assists'] / (df['perf_before_minutes'] / 90)).replace([np.inf, -np.inf], 0).fillna(0)
        df['assists_per_90_after'] = (df['perf_after_assists'] / (df['perf_after_minutes'] / 90)).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Goal contribution (goals + assists)
        df['goal_contribution_before'] = df['perf_before_goals'] + df['perf_before_assists']
        df['goal_contribution_after'] = df['perf_after_goals'] + df['perf_after_assists']
        df['goal_contribution_change'] = df['goal_contribution_after'] - df['goal_contribution_before']
        
        # Minutes per match (playing time consistency)
        df['minutes_per_match_before'] = (df['perf_before_minutes'] / df['perf_before_matches']).fillna(0)
        df['minutes_per_match_after'] = (df['perf_after_minutes'] / df['perf_after_matches']).fillna(0)
        
        return df
    
    def create_player_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on player profile
        """
        logger.info("Creating player profile features...")
        
        # Age-based features
        df['is_young'] = (df['age'] < 23).astype(int)
        df['is_prime'] = ((df['age'] >= 23) & (df['age'] <= 28)).astype(int)
        df['is_veteran'] = (df['age'] > 28).astype(int)
        
        # Position-based features (already have position_group)
        df['is_forward'] = (df['position_group'] == 'Forward').astype(int)
        df['is_midfielder'] = (df['position_group'] == 'Midfielder').astype(int)
        df['is_defender'] = (df['position_group'] == 'Defender').astype(int)
        df['is_goalkeeper'] = (df['position_group'] == 'Goalkeeper').astype(int)
        
        return df
    
    def create_transfer_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on transfer context
        """
        logger.info("Creating transfer context features...")
        
        # Fee-based features
        df['has_fee'] = df['fee_cleaned'].notna().astype(int)
        df['fee_log'] = np.log1p(df['fee_cleaned'].fillna(0))
        
        # Fee categories (already created in EDA)
        # Create binary features for each category
        if 'fee_category' in df.columns:
            df['is_budget_transfer'] = (df['fee_category'] == '<5M').astype(int)
            df['is_mid_transfer'] = (df['fee_category'].isin(['5-15M', '15-30M'])).astype(int)
            df['is_premium_transfer'] = (df['fee_category'] == '>30M').astype(int)
        
        # League features
        league_dummies = pd.get_dummies(df['league_name'], prefix='league')
        df = pd.concat([df, league_dummies], axis=1)
        
        return df
    
    def create_performance_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features comparing player to position/league averages
        """
        logger.info("Creating performance baseline features...")
        
        # Position average performance
        position_avg = df.groupby('position_group').agg({
            'perf_before_goals': 'mean',
            'perf_before_assists': 'mean',
            'perf_before_minutes': 'mean'
        }).add_suffix('_position_avg')
        
        df = df.merge(position_avg, left_on='position_group', right_index=True, how='left')
        
        # Performance relative to position average
        df['goals_vs_position_avg'] = df['perf_before_goals'] - df['perf_before_goals_position_avg']
        df['assists_vs_position_avg'] = df['perf_before_assists'] - df['perf_before_assists_position_avg']
        
        # League average performance
        league_avg = df.groupby('league_name').agg({
            'perf_before_goals': 'mean',
            'perf_before_assists': 'mean'
        }).add_suffix('_league_avg')
        
        df = df.merge(league_avg, left_on='league_name', right_index=True, how='left')
        
        df['goals_vs_league_avg'] = df['perf_before_goals'] - df['perf_before_goals_league_avg']
        df['assists_vs_league_avg'] = df['perf_before_assists'] - df['perf_before_assists_league_avg']
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for different prediction tasks
        """
        logger.info("Creating target variables...")
        
        # Regression targets
        df['target_goals_after'] = df['perf_after_goals']
        df['target_assists_after'] = df['perf_after_assists']
        df['target_goal_contribution_after'] = df['goal_contribution_after']
        df['target_minutes_after'] = df['perf_after_minutes']
        
        # Classification targets
        # Success definition 1: Goal improvement
        df['target_success_goals'] = (df['goal_change'] > 0).astype(int)
        
        # Success definition 2: Minutes increase
        df['target_success_minutes'] = (df['minutes_change'] > 0).astype(int)
        
        # Success definition 3: Composite (goals stable + significant minutes)
        df['target_success_composite'] = ((df['goal_change'] >= 0) & 
                                          (df['minutes_change'] > 500)).astype(int)
        
        # Success definition 4: Goal contribution improvement
        df['target_success_contribution'] = (df['goal_contribution_change'] > 0).astype(int)
        
        return df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """
        Main function to create all features
        """
        logger.info("="*70)
        logger.info("STARTING FEATURE ENGINEERING")
        logger.info("="*70)
        
        # Load base data
        df = self.load_base_data()
        
        # Load extended FBref data
        perf_df = self.load_fbref_extended()
        
        # Create features
        df = self.create_performance_trend_features(df, perf_df)
        df = self.create_player_profile_features(df)
        df = self.create_transfer_context_features(df)
        df = self.create_performance_baseline_features(df)
        df = self.create_target_variables(df)
        
        logger.info(f"\n{'='*70}")
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"Total records: {len(df)}")
        
        # Save engineered features
        output_file = f'{self.output_dir}/transfers_ml_ready.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"\n✅ Saved ML-ready dataset to: {output_file}")
        
        # Print feature summary
        logger.info(f"\nFeature Categories:")
        logger.info(f"  Performance Trend: goals_per_90, assists_per_90, goal_contribution")
        logger.info(f"  Player Profile: is_young, is_prime, is_veteran, position flags")
        logger.info(f"  Transfer Context: fee features, league dummies")
        logger.info(f"  Baseline Comparison: vs position avg, vs league avg")
        logger.info(f"  Target Variables: 4 classification + 4 regression targets")
        
        return df


def main():
    """Main execution"""
    engineer = TransferFeatureEngineer()
    df = engineer.engineer_all_features()
    
    # Display feature summary
    print(f"\n{'='*70}")
    print("FEATURE SUMMARY")
    print(f"{'='*70}")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nSample features:")
    feature_cols = [col for col in df.columns if col.startswith(('goals_per_90', 'is_', 'target_'))]
    print(df[feature_cols[:15]].head().to_string())
    
    print(f"\n{'='*70}")
    print("READY FOR MODEL TRAINING!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

