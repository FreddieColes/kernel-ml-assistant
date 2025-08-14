#!/usr/bin/env python3

import pandas as pd
import json
import mysql.connector
from pathlib import Path
from datetime import datetime

class Phase1Runner:
    """
    This script runs the prepare_dataset pipeline:
    1. Loads the complete patch database.
    2. Filters for scheduling patches in memory.
    3. Extracts features and prepares data for ML.
    4. Exports the results to CSV files.
    """
    def __init__(self):
        self.output_dir = Path('./phase1_output')
        self.output_dir.mkdir(exist_ok=True)
        print(f"Output will be saved to: {self.output_dir}")

    def run_pipeline(self):
        """Executes the complete prepare_dataset data processing pipeline."""
        print("\n--- Starting prepare_dataset Pipeline ---")
        try:
            #Step 1: Load all data from the database
            df_patches = self._load_full_database()
            if df_patches is None or df_patches.empty:
                print("No data loaded. Exiting.")
                return

            #Step 2: Filter for scheduling-related patches
            scheduling_patches = self._filter_scheduling_patches(df_patches)
            
            #Step 3: Extract features from the filtered data
            features_df = self._extract_features(scheduling_patches)
            
            #Step 4: Prepare the data for machine learning models
            X, y = self._prepare_ml_data(features_df)
            
            #Step 5: Generate a summary report and export all data
            self._generate_report(scheduling_patches, features_df)
            self._export_data(scheduling_patches, features_df, X, y)
            
            print("\nPipeline finished successfully.")

        except Exception as e:
            print(f"\n--- An error occurred during pipeline execution ---")
            print(f"Error: {e}")

    def _load_full_database(self):
        """Loads all patches from the 'patch' table in the database."""
        print("Step 1: Loading full database...")
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)['mysql_config']
            config['database'] = 'kernel_patch2'
            
            conn = mysql.connector.connect(**config)
            query = """
                SELECT p.patchwork_id, p.subject, p.author_email, p.send_time,
                       CASE WHEN c.patchwork_id IS NOT NULL THEN 1 ELSE 0 END as accepted
                FROM patch p
                LEFT JOIN commit_to_patch c ON p.patchwork_id = c.patchwork_id
                WHERE p.subject IS NOT NULL
            """
            df = pd.read_sql(query, conn)
            conn.close()
            print(f"-> Loaded {len(df):,} patches.")
            return df
        except Exception as e:
            print(f"Database loading failed: {e}")
            return None

    def _filter_scheduling_patches(self, df_patches):
        """Filters a DataFrame for scheduling-related patches using keywords."""
        print("Step 2: Filtering for scheduling patches...")
        keywords = [
            'sched', 'scheduler', 'scheduling', 'cfs', 'deadline', 'rt',
            'load.balance', 'migration', 'runqueue', 'rq', 'task.group', 
            'cgroup', 'numa', 'preempt', 'fair'
        ]
        pattern = '|'.join(keywords)
        mask = df_patches['subject'].str.contains(pattern, case=False, na=False, regex=True)
        scheduling_patches = df_patches[mask].copy()
        print(f"-> Found {len(scheduling_patches):,} scheduling-related patches.")
        return scheduling_patches

    def _extract_features(self, df):
        """Extracts features from the patch data, matching the original script's logic."""
        print("Step 3: Extracting features...")
        features_df = df.copy()

        #Text features
        if 'subject' in features_df.columns:
            features_df['subject_length'] = features_df['subject'].str.len()
            features_df['word_count'] = features_df['subject'].str.split().str.len()
            features_df['has_fix'] = features_df['subject'].str.contains('fix|bug|error', case=False, na=False)
            features_df['has_performance'] = features_df['subject'].str.contains('performance|optimize|improve', case=False, na=False)
            features_df['has_latency'] = features_df['subject'].str.contains('latency|delay|response', case=False, na=False)
            features_df['has_numa'] = features_df['subject'].str.contains('numa|node', case=False, na=False)
            features_df['has_realtime'] = features_df['subject'].str.contains('rt|real.?time|deadline', case=False, na=False)
            features_df['has_cfs'] = features_df['subject'].str.contains('cfs|fair', case=False, na=False)
            features_df['has_load_balance'] = features_df['subject'].str.contains('load.?balanc|migration', case=False, na=False)
            features_df['has_runqueue'] = features_df['subject'].str.contains('runqueue|rq', case=False, na=False)

        #Author features
        if 'author_email' in features_df.columns:
            author_counts = features_df['author_email'].value_counts()
            features_df['author_patch_count'] = features_df['author_email'].map(author_counts)
            features_df['is_frequent_author'] = features_df['author_patch_count'] >= 10
            scheduler_maintainers = {
                'mingo@kernel.org', 'peterz@infradead.org', 'tglx@linutronix.de',
                'rostedt@goodmis.org', 'bristot@kernel.org'
            }
            features_df['is_scheduler_maintainer'] = features_df['author_email'].isin(scheduler_maintainers)

        #Temporal features
        if 'send_time' in features_df.columns:
            features_df['timestamp'] = pd.to_datetime(features_df['send_time'], unit='ms', errors='coerce')
            features_df['hour'] = features_df['timestamp'].dt.hour
            features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
            features_df['month'] = features_df['timestamp'].dt.month
            features_df['year'] = features_df['timestamp'].dt.year
            features_df['is_business_hours'] = (features_df['hour'] >= 9) & (features_df['hour'] <= 17)
            features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6])
        
        print(f"-> Extracted {len(features_df.columns)} total features.")
        return features_df

    def _prepare_ml_data(self, features_df):
        """Prepares DataFrames X (features) and y (target) for ML."""
        print("Step 4: Preparing ML-ready datasets...")
        if features_df is None or features_df.empty:
            return None, None
            
        #Select numerical and boolean features, excluding identifiers and the target
        feature_cols = features_df.select_dtypes(include=['number', 'bool']).columns.tolist()
        exclude_cols = {'accepted', 'patchwork_id', 'send_time'}
        ml_features = [col for col in feature_cols if col not in exclude_cols]
        
        X = features_df[ml_features].fillna(0)
        y = features_df['accepted']
        
        print(f"-> Prepared ML data with {len(X):,} samples and {len(X.columns)} features.")
        return X, y

    def _generate_report(self, patches_df, features_df):
        """Generates and saves a JSON summary report."""
        print("Step 5.1: Generating analysis report...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f'report_{timestamp}.json'
        
        report_data = {
            'timestamp': timestamp,
            'total_scheduling_patches': len(patches_df),
            'features_extracted': len(features_df.columns),
            'acceptance_rate': patches_df['accepted'].mean() if 'accepted' in patches_df else 0,
            'yearly_distribution': features_df['year'].value_counts().sort_index().to_dict() if 'year' in features_df else {}
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=4, default=str)
        
        print(f"-> Saved analysis report: {report_file}")

    def _export_data(self, patches_df, features_df, X, y):
        """Exports the processed DataFrames to CSV files."""
        print("Step 5.2: Exporting data to CSV files...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        patches_df.to_csv(self.output_dir / f'patches_{timestamp}.csv', index=False)
        features_df.to_csv(self.output_dir / f'features_{timestamp}.csv', index=False)
        if X is not None and y is not None:
            X.to_csv(self.output_dir / f'X_features_{timestamp}.csv', index=False)
            y.to_csv(self.output_dir / f'y_target_{timestamp}.csv', index=False)
        
        print(f"-> Data exported successfully.")


if __name__ == "__main__":
    runner = Phase1Runner()
    runner.run_pipeline()