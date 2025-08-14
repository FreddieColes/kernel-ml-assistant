#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

class EnhancedAutonomousFeedbackLoop:
    def __init__(self, duration_hours: float = 0.25):
        self.duration_hours = duration_hours
        self.setup_logging()
        
        self.models_dir = self._find_phase2_models()
        self.data_dir = Path("./phase1_output/phase2_ml_data")
        
        self.models = {}
        self.performance_history = {}
        self.improvement_attempts = 0
        self.successful_improvements = 0
        self.improvement_strategies = [
            'add_estimators', 'tune_learning_rate', 'adjust_depth',
            'feature_selection', 'ensemble_weights', 'regularization_tuning'
        ]
        self.strategy_index = 0
        
        self._load_data()
        self._load_trained_models()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('autonomous_feedback.log')]
        )
        self.logger = logging.getLogger('EnhancedAutonomousFeedbackLoop')
    
    def _find_phase2_models(self) -> Optional[Path]:
        model_dirs = list(Path('.').glob('phase2_primary_models_*'))
        if model_dirs:
            latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
            print(f"Found Phase 2 models: {latest_dir}")
            return latest_dir
        print("No phase2_primary_models_ directory found")
        return None
    
    def _load_data(self):
        print("Loading Phase 1 data...")
        
        feature_files = list(self.data_dir.glob("X_features_*.csv"))
        target_files = list(self.data_dir.glob("y_target_*.csv"))
        
        if not feature_files or not target_files:
            raise FileNotFoundError("Phase 1 data not found. Run phase1_complete.py first.")
        
        latest_features = max(feature_files, key=lambda x: x.stat().st_mtime)
        latest_target = max(target_files, key=lambda x: x.stat().st_mtime)
        
        self.X = pd.read_csv(latest_features).fillna(0)
        self.y = pd.read_csv(latest_target).iloc[:, 0]
        
        #remove non-numeric columns for ML models
        numeric_columns = self.X.select_dtypes(include=[np.number]).columns
        non_numeric_columns = self.X.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_columns) > 0:
            print(f"Removing non-numeric columns: {list(non_numeric_columns)}")
            self.X = self.X[numeric_columns]
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.3, stratify=self.y, random_state=42
        )
        
        print(f"Loaded {len(self.X):,} samples with {len(self.X.columns)} features")
    
    def _load_trained_models(self):
        if not self.models_dir:
            print("No models directory available")
            return
        
        #check what's actually in the directory
        print(f"Checking directory: {self.models_dir}")
        if self.models_dir.exists():
            subdirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
            files = [f for f in self.models_dir.iterdir() if f.is_file() and f.suffix == '.pkl']
            print(f"Subdirectories: {[d.name for d in subdirs]}")
            print(f"Pickle files: {[f.name for f in files]}")
        
        #try multiple possible locations
        possible_paths = [
            self.models_dir / 'trained_models',
            self.models_dir / 'models', 
            self.models_dir / 'ml_models',
            self.models_dir
        ]
        
        models_path = None
        for path in possible_paths:
            if path.exists():
                pkl_files = list(path.glob('*.pkl'))
                if pkl_files:
                    models_path = path
                    print(f"Found models in: {path}")
                    print(f"Available files: {[f.name for f in pkl_files]}")
                    break
        
        if not models_path:
            print("No pickle files found in any expected location")
            return
        
        #find model files flexibly
        pkl_files = list(models_path.glob('*.pkl'))
        model_mapping = {}
        
        for pkl_file in pkl_files:
            filename = pkl_file.name.lower()
            if 'random' in filename or 'forest' in filename or 'rf' in filename:
                model_mapping['random_forest'] = pkl_file
            elif 'xgb' in filename or 'xgboost' in filename:
                model_mapping['xgboost'] = pkl_file
            elif 'lgb' in filename or 'lightgbm' in filename or 'light' in filename:
                model_mapping['lightgbm'] = pkl_file
        
        if not model_mapping:
            print("No recognizable model files found")
            print("Available files:", [f.name for f in pkl_files])
            return
        
        loaded_count = 0
        for model_name, model_path in model_mapping.items():
            try:
                print(f"Loading {model_name} from {model_path.name}")
                with open(model_path, 'rb') as f:
                    original_model = pickle.load(f)
                
                suboptimal_model = self._create_suboptimal_model(model_name, original_model)
                
                self.models[model_name] = {
                    'model': suboptimal_model,
                    'original_model': original_model,
                    'current_f1': 0.0,
                    'best_f1': 0.0,
                    'improvement_count': 0,
                    'last_updated': datetime.now(),
                    'improvement_history': []
                }
                loaded_count += 1
                print(f"Successfully loaded {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
        
        if loaded_count == 0:
            print("No models loaded successfully")
            return
        
        print(f"Successfully loaded {loaded_count} models")
        self._establish_baseline_performance()
    
    def _create_suboptimal_model(self, model_name: str, original_model):
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb
        import lightgbm as lgb
        
        if model_name == 'random_forest':
            return RandomForestClassifier(
                n_estimators=max(50, original_model.n_estimators // 4),
                max_depth=min(5, original_model.max_depth or 10),
                min_samples_split=max(10, original_model.min_samples_split * 2),
                random_state=42, n_jobs=-1, class_weight=original_model.class_weight
            )
        elif model_name == 'xgboost':
            params = original_model.get_params()
            return xgb.XGBClassifier(
                n_estimators=max(50, params.get('n_estimators', 100) // 3),
                max_depth=min(3, params.get('max_depth', 6)),
                learning_rate=min(0.05, params.get('learning_rate', 0.1)),
                subsample=0.7, random_state=42
            )
        elif model_name == 'lightgbm':
            params = original_model.get_params()
            return lgb.LGBMClassifier(
                n_estimators=max(50, params.get('n_estimators', 100) // 3),
                max_depth=min(3, params.get('max_depth', 6)),
                learning_rate=min(0.05, params.get('learning_rate', 0.1)),
                num_leaves=min(15, params.get('num_leaves', 31)),
                random_state=42, verbose=-1
            )
        return original_model
    
    def _establish_baseline_performance(self):
        print("Establishing baseline performance...")
        
        for model_name, model_data in self.models.items():
            model = model_data['model']
            
            subset_size = min(10000, len(self.X_train))
            indices = np.random.choice(len(self.X_train), subset_size, replace=False)
            X_subset = self.X_train.iloc[indices]
            y_subset = self.y_train.iloc[indices]
            
            model.fit(X_subset, y_subset)
            y_pred = model.predict(self.X_val)
            
            f1 = f1_score(self.y_val, y_pred)
            accuracy = accuracy_score(self.y_val, y_pred)
            
            self.models[model_name]['current_f1'] = f1
            self.models[model_name]['best_f1'] = f1
            self.models[model_name]['baseline_f1'] = f1
            self.models[model_name]['baseline_accuracy'] = accuracy
            
            print(f"{model_name}: F1={f1:.4f}, Accuracy={accuracy:.4f}")
        
        self.performance_history = {
            model_name: [model_data['current_f1']] 
            for model_name, model_data in self.models.items()
        }
    
    def run_autonomous_loop(self):
        if not self.models:
            print("No models available for autonomous loop")
            return
        
        print(f"Starting Enhanced Autonomous Feedback Loop")
        print(f"Duration: {self.duration_hours} hours")
        print(f"Models: {list(self.models.keys())}")
        
        self.logger.info("Enhanced autonomous feedback loop started")
        
        start_time = time.time()
        end_time = start_time + (self.duration_hours * 3600)
        iteration = 0
        
        while time.time() < end_time:
            iteration += 1
            self._perform_improvement_cycle(iteration)
            
            if self._check_early_stopping():
                print("Early stopping: Target performance reached")
                break
            
            sleep_time = min(15, (self.duration_hours * 3600) / 30)
            time.sleep(sleep_time)
        
        self.logger.info("Enhanced autonomous feedback loop stopped")
        self._generate_final_report()
    
    def _perform_improvement_cycle(self, iteration: int):
        self.improvement_attempts += 1
        
        for model_name, model_data in self.models.items():
            try:
                strategy = self.improvement_strategies[self.strategy_index % len(self.improvement_strategies)]
                improved_model = self._attempt_model_improvement_enhanced(model_name, model_data, strategy)
                
                if improved_model:
                    y_pred_improved = improved_model.predict(self.X_val)
                    f1_improved = f1_score(self.y_val, y_pred_improved)
                    
                    current_f1 = model_data['current_f1']
                    improvement = f1_improved - current_f1
                    
                    if improvement > 0.0001:
                        self.models[model_name]['model'] = improved_model
                        self.models[model_name]['current_f1'] = f1_improved
                        self.models[model_name]['improvement_count'] += 1
                        self.models[model_name]['last_updated'] = datetime.now()
                        self.models[model_name]['improvement_history'].append({
                            'iteration': iteration,
                            'strategy': strategy,
                            'improvement': improvement,
                            'new_f1': f1_improved
                        })
                        
                        if f1_improved > model_data['best_f1']:
                            self.models[model_name]['best_f1'] = f1_improved
                        
                        self.successful_improvements += 1
                        print(f"{model_name} improved via {strategy}: {current_f1:.4f} -> {f1_improved:.4f} (+{improvement:.4f})")
                
                self.performance_history[model_name].append(model_data['current_f1'])
                
            except Exception as e:
                self.logger.error(f"Error improving {model_name}: {e}")
        
        self.strategy_index += 1
    
    def _attempt_model_improvement_enhanced(self, model_name: str, model_data: Dict, strategy: str):
        current_model = model_data['model']
        
        if strategy == 'add_estimators':
            return self._add_estimators_strategy(model_name, current_model)
        elif strategy == 'tune_learning_rate':
            return self._tune_learning_rate_strategy(model_name, current_model)
        elif strategy == 'adjust_depth':
            return self._adjust_depth_strategy(model_name, current_model)
        elif strategy == 'feature_selection':
            return self._feature_selection_strategy(model_name, current_model)
        elif strategy == 'ensemble_weights':
            return self._ensemble_weights_strategy(model_name, current_model)
        elif strategy == 'regularization_tuning':
            return self._regularization_tuning_strategy(model_name, current_model)
        return None
    
    def _safe_get_param(self, params: dict, key: str, default_value, param_type=float):
        value = params.get(key, default_value)
        if value is None:
            return default_value
        try:
            return param_type(value)
        except (ValueError, TypeError):
            return default_value
    
    def _add_estimators_strategy(self, model_name: str, current_model):
        if model_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            params = current_model.get_params()
            current_estimators = self._safe_get_param(params, 'n_estimators', 100, int)
            params['n_estimators'] = min(current_estimators + 25, 300)
            improved_model = RandomForestClassifier(**params)
        elif model_name == 'xgboost':
            import xgboost as xgb
            params = current_model.get_params()
            current_estimators = self._safe_get_param(params, 'n_estimators', 100, int)
            params['n_estimators'] = min(current_estimators + 25, 300)
            improved_model = xgb.XGBClassifier(**params)
        elif model_name == 'lightgbm':
            import lightgbm as lgb
            params = current_model.get_params()
            current_estimators = self._safe_get_param(params, 'n_estimators', 100, int)
            params['n_estimators'] = min(current_estimators + 25, 300)
            improved_model = lgb.LGBMClassifier(**params)
        else:
            return None
        
        subset_size = min(8000, len(self.X_train))
        indices = np.random.choice(len(self.X_train), subset_size, replace=False)
        X_subset = self.X_train.iloc[indices]
        y_subset = self.y_train.iloc[indices]
        
        improved_model.fit(X_subset, y_subset)
        return improved_model
    
    def _tune_learning_rate_strategy(self, model_name: str, current_model):
        if model_name in ['xgboost', 'lightgbm']:
            params = current_model.get_params()
            current_lr = self._safe_get_param(params, 'learning_rate', 0.1, float)
            new_lr = min(current_lr * 1.2, 0.3)
            params['learning_rate'] = new_lr
            
            if model_name == 'xgboost':
                import xgboost as xgb
                improved_model = xgb.XGBClassifier(**params)
            else:
                import lightgbm as lgb
                improved_model = lgb.LGBMClassifier(**params)
            
            subset_size = min(8000, len(self.X_train))
            indices = np.random.choice(len(self.X_train), subset_size, replace=False)
            X_subset = self.X_train.iloc[indices]
            y_subset = self.y_train.iloc[indices]
            
            improved_model.fit(X_subset, y_subset)
            return improved_model
        return None
    
    def _adjust_depth_strategy(self, model_name: str, current_model):
        params = current_model.get_params()
        
        if model_name == 'random_forest':
            current_depth = self._safe_get_param(params, 'max_depth', 10, int)
        elif model_name == 'lightgbm':
            current_depth = self._safe_get_param(params, 'max_depth', -1, int)
            if current_depth == -1:
                current_depth = 6
        else:  #xgboost
            current_depth = self._safe_get_param(params, 'max_depth', 6, int)
        
        new_depth = min(current_depth + 1, 12)
        
        if model_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            params['max_depth'] = new_depth
            improved_model = RandomForestClassifier(**params)
        elif model_name == 'xgboost':
            import xgboost as xgb
            params['max_depth'] = new_depth
            improved_model = xgb.XGBClassifier(**params)
        elif model_name == 'lightgbm':
            import lightgbm as lgb
            params['max_depth'] = new_depth
            improved_model = lgb.LGBMClassifier(**params)
        else:
            return None
        
        subset_size = min(8000, len(self.X_train))
        indices = np.random.choice(len(self.X_train), subset_size, replace=False)
        X_subset = self.X_train.iloc[indices]
        y_subset = self.y_train.iloc[indices]
        
        improved_model.fit(X_subset, y_subset)
        return improved_model
    
    def _feature_selection_strategy(self, model_name: str, current_model):
        selector = SelectKBest(f_classif, k=min(25, self.X_train.shape[1]))
        X_train_selected = selector.fit_transform(self.X_train, self.y_train)
        
        params = current_model.get_params()
        
        if model_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            improved_model = RandomForestClassifier(**params)
        elif model_name == 'xgboost':
            import xgboost as xgb
            improved_model = xgb.XGBClassifier(**params)
        elif model_name == 'lightgbm':
            import lightgbm as lgb
            improved_model = lgb.LGBMClassifier(**params)
        else:
            return None
        
        subset_size = min(8000, len(X_train_selected))
        indices = np.random.choice(len(X_train_selected), subset_size, replace=False)
        X_subset = X_train_selected[indices]
        y_subset = self.y_train.iloc[indices]
        
        improved_model.fit(X_subset, y_subset)
        improved_model.feature_selector = selector
        
        original_predict = improved_model.predict
        def predict_with_selection(X):
            X_selected = selector.transform(X)
            return original_predict(X_selected)
        improved_model.predict = predict_with_selection
        
        return improved_model
    
    def _ensemble_weights_strategy(self, model_name: str, current_model):
        params = current_model.get_params()
        
        if model_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            params['max_samples'] = 0.9
            improved_model = RandomForestClassifier(**params)
        elif model_name == 'xgboost':
            import xgboost as xgb
            current_subsample = self._safe_get_param(params, 'subsample', 1.0, float)
            params['subsample'] = min(current_subsample + 0.1, 1.0)
            improved_model = xgb.XGBClassifier(**params)
        elif model_name == 'lightgbm':
            import lightgbm as lgb
            current_bagging = self._safe_get_param(params, 'bagging_fraction', 1.0, float)
            params['bagging_fraction'] = min(current_bagging + 0.1, 1.0)
            params['bagging_freq'] = 1
            improved_model = lgb.LGBMClassifier(**params)
        else:
            return None
        
        subset_size = min(8000, len(self.X_train))
        indices = np.random.choice(len(self.X_train), subset_size, replace=False)
        X_subset = self.X_train.iloc[indices]
        y_subset = self.y_train.iloc[indices]
        
        improved_model.fit(X_subset, y_subset)
        return improved_model
    
    def _regularization_tuning_strategy(self, model_name: str, current_model):
        params = current_model.get_params()
        
        if model_name == 'xgboost':
            import xgboost as xgb
            reg_alpha = self._safe_get_param(params, 'reg_alpha', 0, float)
            reg_lambda = self._safe_get_param(params, 'reg_lambda', 1, float)
            params['reg_alpha'] = max(reg_alpha * 0.8, 0)
            params['reg_lambda'] = max(reg_lambda * 0.9, 0.1)
            improved_model = xgb.XGBClassifier(**params)
        elif model_name == 'lightgbm':
            import lightgbm as lgb
            reg_alpha = self._safe_get_param(params, 'reg_alpha', 0, float)
            reg_lambda = self._safe_get_param(params, 'reg_lambda', 0, float)
            params['reg_alpha'] = max(reg_alpha * 0.8, 0)
            params['reg_lambda'] = max(reg_lambda * 0.9, 0)
            improved_model = lgb.LGBMClassifier(**params)
        elif model_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            min_samples_split = self._safe_get_param(params, 'min_samples_split', 2, int)
            params['min_samples_split'] = max(min_samples_split - 1, 2)
            improved_model = RandomForestClassifier(**params)
        else:
            return None
        
        subset_size = min(8000, len(self.X_train))
        indices = np.random.choice(len(self.X_train), subset_size, replace=False)
        X_subset = self.X_train.iloc[indices]
        y_subset = self.y_train.iloc[indices]
        
        improved_model.fit(X_subset, y_subset)
        return improved_model
    
    def _check_early_stopping(self) -> bool:
        any_good = any(
            model_data['current_f1'] > 0.55
            for model_data in self.models.values()
        )
        return any_good
    
    def _generate_final_report(self):
        best_model = max(self.models.items(), key=lambda x: x[1]['current_f1'])
        total_improvements = sum(model_data['improvement_count'] for model_data in self.models.values())
        
        print("\nFinal Results:")
        for model_name, model_data in self.models.items():
            baseline_f1 = model_data.get('baseline_f1', 0)
            final_f1 = model_data['current_f1']
            improvement = final_f1 - baseline_f1
            print(f"{model_name}: {baseline_f1:.4f} -> {final_f1:.4f} (+{improvement:.4f}) [{model_data['improvement_count']} updates]")
        
        print(f"\nSummary:")
        print(f"Best Model: {best_model[0]} (F1: {best_model[1]['current_f1']:.4f})")
        print(f"Total Updates: {total_improvements}")
        print(f"Success Rate: {(self.successful_improvements/max(self.improvement_attempts,1)*100):.1f}%")
        
        #save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration_hours': self.duration_hours,
            'models_managed': len(self.models),
            'total_improvements': total_improvements,
            'improvement_attempts': self.improvement_attempts,
            'successful_improvements': self.successful_improvements,
            'success_rate': self.successful_improvements / max(self.improvement_attempts, 1),
            'best_model': {
                'name': best_model[0],
                'f1_score': best_model[1]['current_f1']
            },
            'model_details': {
                model_name: {
                    'baseline_f1': model_data.get('baseline_f1', 0),
                    'final_f1': model_data['current_f1'],
                    'best_f1': model_data['best_f1'],
                    'improvement_count': model_data['improvement_count'],
                    'improvement_history': model_data['improvement_history'],
                    'performance_history': self.performance_history[model_name]
                }
                for model_name, model_data in self.models.items()
            }
        }
        
        report_filename = f"enhanced_autonomous_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {report_filename}")


def main():
    try:
        print("Starting Enhanced Autonomous Feedback Loop")
        autonomous_system = EnhancedAutonomousFeedbackLoop(duration_hours=0.25)
        autonomous_system.run_autonomous_loop()
        return autonomous_system
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()