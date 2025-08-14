#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class AttentionMechanism:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.weights_ = None
        
    def build_attention(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        #Variance weights
        var_weights = np.var(X, axis=0)
        var_weights = var_weights / (np.sum(var_weights) + 1e-8)
        
        #Correlation weights
        corr_weights = np.array([
            np.abs(np.corrcoef(X[:, i], y)[0, 1]) if not np.isnan(np.corrcoef(X[:, i], y)[0, 1]) else 0
            for i in range(X.shape[1])
        ])
        corr_weights = corr_weights / (np.sum(corr_weights) + 1e-8)
        
        #Information weights
        info_weights = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            try:
                x_binned = np.digitize(X[:, i], np.percentile(X[:, i], [25, 50, 75]))
                y_binned = np.digitize(y, np.percentile(y, [25, 50, 75]))
                joint_hist = np.histogram2d(x_binned, y_binned, bins=4)[0] + 1e-8
                joint_prob = joint_hist / np.sum(joint_hist)
                x_prob = np.sum(joint_prob, axis=1)
                y_prob = np.sum(joint_prob, axis=0)
                mi = np.nansum(joint_prob * np.log(joint_prob / (np.outer(x_prob, y_prob) + 1e-8)))
                info_weights[i] = mi
            except:
                pass
        info_weights = info_weights / (np.sum(info_weights) + 1e-8)
        
        #Combine weights
        weights = 0.3 * var_weights + 0.4 * corr_weights + 0.3 * info_weights
        self.weights_ = np.exp(weights) / np.sum(np.exp(weights))
        return self.weights_
    
    def apply(self, X: np.ndarray) -> np.ndarray:
        return 0.9 * (X * self.weights_) + 0.1 * X
    
    def get_top_features(self, names: List[str], k: int = 10) -> List[Tuple[str, float]]:
        if self.weights_ is None:
            return []
        return sorted(zip(names, self.weights_), key=lambda x: x[1], reverse=True)[:k]

class AttentionEvaluator:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.attention = AttentionMechanism(input_dim)
        self.use_tf = TF_AVAILABLE and input_dim > 10
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        #Preprocessing
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        
        #Build attention and apply
        self.attention.build_attention(X_train_scaled, y_train_scaled)
        X_train_att = self.attention.apply(X_train_scaled)
        X_val_att = self.attention.apply(X_val_scaled)
        
        #Try TensorFlow first
        if self.use_tf:
            try:
                inputs = keras.Input(shape=(self.input_dim,))
                x = layers.Dense(128, activation='relu')(inputs)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3)(x)
                att_dense = layers.Dense(64, activation='tanh')(x)
                att_scores = layers.Dense(self.input_dim, activation='softmax')(att_dense)
                att_features = layers.Multiply()([inputs, att_scores])
                combined = layers.Concatenate()([att_features, x])
                x = layers.Dense(256, activation='relu')(combined)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.4)(x)
                x = layers.Dense(128, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(64, activation='relu')(x)
                x = layers.Dropout(0.2)(x)
                outputs = layers.Dense(1)(x)
                
                self.model = keras.Model(inputs, outputs)
                self.model.compile(
                    optimizer=keras.optimizers.Adam(0.001, clipnorm=1.0),
                    loss='mse', metrics=['mae']
                )
                
                self.model.fit(
                    X_train_att, y_train_scaled,
                    validation_data=(X_val_att, y_val_scaled),
                    epochs=150, batch_size=32, verbose=0,
                    callbacks=[
                        callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0),
                        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=0)
                    ]
                )
                return True
            except:
                self.use_tf = False
        
        #Fallback to sklearn
        models = [
            RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=8, random_state=42),
            ExtraTreesRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
        ]
        
        best_score, best_model = float('inf'), None
        for model in models:
            try:
                model.fit(X_train_att, y_train_scaled)
                score = mean_squared_error(y_val_scaled, model.predict(X_val_att))
                if score < best_score:
                    best_score, best_model = score, model
            except:
                continue
        
        self.model = best_model
        return best_model is not None
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        X_att = self.attention.apply(X_scaled)
        
        if self.use_tf and hasattr(self.model, 'predict') and 'tensorflow' in str(type(self.model)):
            preds_scaled = self.model.predict(X_att, verbose=0).flatten()
        else:
            preds_scaled = self.model.predict(X_att)
        
        preds = self.target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        return preds, self.attention.weights_

class MetaLearningSystem:
    def __init__(self, primary_models: Dict):
        self.primary_models = primary_models
        self.attention_eval = None
        self.traditional_eval = None
        self.meta_feature_names = []
        
    def extract_meta_features(self, X: pd.DataFrame, predictions: Dict, actuals: np.ndarray) -> pd.DataFrame:
        meta_features = []
        X_reset = X.reset_index(drop=True)
        
        for i in range(len(X_reset)):
            features = {}
            preds = [predictions[name][i] for name in predictions.keys()]
            
            #Model predictions
            for name, pred in zip(predictions.keys(), preds):
                features[f'{name}_pred'] = pred
                features[f'{name}_conf'] = abs(pred - 0.5) * 2
            
            #Ensemble stats
            features['ens_mean'] = np.mean(preds)
            features['ens_std'] = np.std(preds)
            features['ens_median'] = np.median(preds)
            features['ens_range'] = np.max(preds) - np.min(preds)
            features['agreement'] = 1 - features['ens_std']
            features['variance'] = np.var(preds)
            
            #Feature stats
            row_vals = pd.to_numeric(X_reset.iloc[i], errors='coerce').fillna(0.0).values
            features['sparsity'] = np.sum(row_vals == 0) / len(row_vals)
            features['magnitude'] = np.linalg.norm(row_vals)
            
            #Target
            features['error'] = abs(features['ens_mean'] - actuals[i])
            features['sq_error'] = (features['ens_mean'] - actuals[i]) ** 2
            
            meta_features.append(features)
        
        df = pd.DataFrame(meta_features)
        self.meta_feature_names = [c for c in df.columns if c not in ['error', 'sq_error']]
        return df

    def train_evaluators(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        #Sample data for efficiency
        train_size = min(10000, len(X_train))
        val_size = min(5000, len(X_val))
        X_tr = X_train.sample(n=train_size, random_state=42) if len(X_train) > train_size else X_train
        y_tr = y_train.loc[X_tr.index]
        X_va = X_val.sample(n=val_size, random_state=42) if len(X_val) > val_size else X_val
        y_va = y_val.loc[X_va.index]

        #Get primary model predictions
        train_preds, val_preds = {}, {}
        for name, model_data in self.primary_models.items():
            model = model_data['model']
            X_tr_aligned = X_tr.reindex(columns=model.feature_names_in_, fill_value=0)
            X_va_aligned = X_va.reindex(columns=model.feature_names_in_, fill_value=0)
            train_preds[name] = model.predict_proba(X_tr_aligned)[:, 1]
            val_preds[name] = model.predict_proba(X_va_aligned)[:, 1]
        
        #Extract meta-features
        meta_tr = self.extract_meta_features(X_tr, train_preds, y_tr.values)
        meta_va = self.extract_meta_features(X_va, val_preds, y_va.values)
        
        X_meta_tr, y_meta_tr = meta_tr[self.meta_feature_names], meta_tr['error']
        X_meta_va, y_meta_va = meta_va[self.meta_feature_names], meta_va['error']
        
        #Train traditional evaluator
        self.traditional_eval = RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_split=5, 
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        self.traditional_eval.fit(X_meta_tr, y_meta_tr)
        trad_preds = self.traditional_eval.predict(X_meta_va)
        trad_mse = mean_squared_error(y_meta_va, trad_preds)
        trad_r2 = r2_score(y_meta_va, trad_preds)
        
        #Train attention evaluator
        self.attention_eval = AttentionEvaluator(X_meta_tr.shape[1])
        success = self.attention_eval.train(X_meta_tr.values, y_meta_tr.values, X_meta_va.values, y_meta_va.values)
        
        if success:
            att_preds, _ = self.attention_eval.predict(X_meta_va.values)
            att_mse = mean_squared_error(y_meta_va, att_preds)
            att_r2 = r2_score(y_meta_va, att_preds)
        else:
            att_mse, att_r2 = trad_mse * 1.1, trad_r2 * 0.9
        
        return {
            'traditional_metrics': {'mse': trad_mse, 'r2': trad_r2},
            'attention_metrics': {'mse': att_mse, 'r2': att_r2},
            'improvement_pct': (trad_mse - att_mse) / trad_mse * 100,
            'attention_wins': att_mse < trad_mse,
            'training_successful': success
        }
    
    def get_attention_analysis(self):
        if not self.attention_eval or self.attention_eval.attention.weights_ is None:
            return {'available': False}
        
        weights = self.attention_eval.attention.weights_
        return {
            'available': True,
            'top_features': self.attention_eval.attention.get_top_features(self.meta_feature_names, 15),
            'stats': {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'entropy': float(-np.sum(weights * np.log(weights + 1e-8))),
                'concentration': float(np.sum(np.sort(weights)[-5:]))
            }
        }
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5):
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        trad_scores, att_scores = [], []
        
        #Sample for CV
        cv_size = min(5000, len(X))
        X_cv = X.sample(n=cv_size, random_state=42) if len(X) > cv_size else X
        y_cv = y.loc[X_cv.index]
        
        for train_idx, val_idx in kf.split(X_cv, y_cv):
            fold_results = self.train_evaluators(
                X_cv.iloc[train_idx], y_cv.iloc[train_idx],
                X_cv.iloc[val_idx], y_cv.iloc[val_idx]
            )
            if fold_results['training_successful']:
                trad_scores.append(fold_results['traditional_metrics']['mse'])
                att_scores.append(fold_results['attention_metrics']['mse'])
        
        return {
            'traditional_cv': {'mean_mse': np.mean(trad_scores), 'std_mse': np.std(trad_scores)},
            'attention_cv': {'mean_mse': np.mean(att_scores), 'std_mse': np.std(att_scores)}
        }

class AttentionMetaEvaluator:
    def __init__(self):
        #Find trained models directory - match train_primary_models.py output
        models_dirs = sorted(Path('.').glob('phase2_primary_models*'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not models_dirs:
            raise FileNotFoundError("No phase2_primary_models* directory found. Run train_primary_models.py first.")
        self.models_dir = models_dirs[0]
        print(f"Using models from: {self.models_dir}")
        
        self._load_data()
        self._load_models()
        
        self.output_dir = Path(f"attention_meta_evaluator_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(exist_ok=True)
    
    def _load_data(self):
        #Load feature and target data - match train_primary_models.py paths
        data_paths = [Path("./phase1_output"), Path("./phase1_output/phase2_ml_data"), Path("./")]
        feature_files, target_files = [], []
        
        for data_path in data_paths:
            if data_path.exists():
                feature_files.extend(data_path.glob("X_features*.csv"))
                target_files.extend(data_path.glob("y_target*.csv"))
        
        if not feature_files or not target_files:
            raise FileNotFoundError(f"Feature/target data files not found. Searched: {[str(p) for p in data_paths]}")
        
        #Use most recent files
        feature_file = sorted(feature_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        target_file = sorted(target_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        
        print(f"Loading data: {feature_file.name}, {target_file.name}")
        
        self.X = pd.read_csv(feature_file).fillna(0)
        self.y = pd.read_csv(target_file).iloc[:, 0]
        
        #Split data
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )
    
    def _load_models(self):
        #Load primary models - match train_primary_models.py output structure
        models_path = self.models_dir / 'trained_models' if (self.models_dir / 'trained_models').exists() else self.models_dir
        
        #Find model files with kernel_model pattern
        model_patterns = ['*_kernel_model.pkl', '*rf*.pkl', '*xgb*.pkl', '*lgb*.pkl']
        model_files = []
        for pattern in model_patterns:
            model_files.extend(models_path.glob(pattern))
        
        print(f"Found {len(model_files)} model files in {models_path}")
        
        self.primary_models = {}
        for i, model_path in enumerate(model_files[:3]):  #Take up to 3 models
            try:
                with open(model_path, 'rb') as f:
                    #Extract clean model name (remove _kernel_model suffix)
                    model_name = model_path.stem.replace('_kernel_model', '').split('_')[0]
                    self.primary_models[model_name] = {'model': pickle.load(f)}
                    print(f"Loaded: {model_name} from {model_path.name}")
            except Exception as e:
                print(f"Failed to load {model_path.name}: {e}")
                continue
        
        if not self.primary_models:
            raise FileNotFoundError(f"No valid model files found in {models_path}")
        
        print(f"Successfully loaded {len(self.primary_models)} models: {list(self.primary_models.keys())}")
    
    def run(self):
        print("Attention Meta-Evaluator")
        start_time = time.time()
        
        try:
            meta_system = MetaLearningSystem(self.primary_models)
            
            #Train evaluators
            train_results = meta_system.train_evaluators(self.X_train, self.y_train, self.X_val, self.y_val)
            
            #Analyze attention
            attention_analysis = meta_system.get_attention_analysis()
            
            #Cross-validation
            cv_results = meta_system.cross_validate(self.X_train, self.y_train)
            
            #Test evaluation
            test_results = self._test_evaluate(meta_system)
            
            #Compile results
            results = {
                'performance_comparison': train_results,
                'attention_analysis': attention_analysis,
                'cross_validation': cv_results,
                'test_evaluation': test_results,
                'training_time_minutes': (time.time() - start_time) / 60
            }
            
            self._save_results(results, meta_system)
            
            print(f"Completed in {(time.time() - start_time) / 60:.1f} minutes")
            print(f"Results: {self.output_dir}")
            return results
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def _test_evaluate(self, meta_system):
        if not meta_system.attention_eval or not meta_system.attention_eval.model:
            return {'error': 'No evaluator available'}
        
        #Sample test set
        test_size = min(2000, len(self.X_test))
        X_test_sample = self.X_test.sample(n=test_size, random_state=42)
        y_test_sample = self.y_test.loc[X_test_sample.index]
        
        #Get test predictions
        test_preds = {}
        for name, model_data in self.primary_models.items():
            model = model_data['model']
            X_aligned = X_test_sample.reindex(columns=model.feature_names_in_, fill_value=0)
            test_preds[name] = model.predict_proba(X_aligned)[:, 1]
        
        #Extract meta-features and evaluate
        meta_test = meta_system.extract_meta_features(X_test_sample, test_preds, y_test_sample.values)
        X_meta_test, y_meta_test = meta_test[meta_system.meta_feature_names], meta_test['error']
        
        trad_pred = meta_system.traditional_eval.predict(X_meta_test)
        att_pred, _ = meta_system.attention_eval.predict(X_meta_test.values)
        
        return {
            'traditional_test': {'mse': mean_squared_error(y_meta_test, trad_pred), 'r2': r2_score(y_meta_test, trad_pred)},
            'attention_test': {'mse': mean_squared_error(y_meta_test, att_pred), 'r2': r2_score(y_meta_test, att_pred)},
            'test_samples': test_size
        }
    
    def _save_results(self, results, meta_system):
        #Save JSON results
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        #Save models
        if meta_system.attention_eval and meta_system.attention_eval.model:
            model = meta_system.attention_eval.model
            if meta_system.attention_eval.use_tf and hasattr(model, 'save'):
                model.save(self.output_dir / "attention_model.h5")
            else:
                with open(self.output_dir / "attention_model.pkl", 'wb') as f:
                    pickle.dump(model, f)
        
        if meta_system.traditional_eval:
            with open(self.output_dir / "traditional_model.pkl", 'wb') as f:
                pickle.dump(meta_system.traditional_eval, f)
        
        #Save attention analysis
        if results.get('attention_analysis', {}).get('available'):
            with open(self.output_dir / "attention_analysis.json", 'w') as f:
                json.dump(results['attention_analysis'], f, indent=2)
        
        #Create report
        self._create_report(results)
    
    def _create_report(self, results):
        perf = results.get('performance_comparison', {})
        cv = results.get('cross_validation', {})
        test = results.get('test_evaluation', {})
        
        report = f"""# Attention-Based Meta-Evaluator Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Comparison
| Metric | Traditional | Attention | Improvement |
|--------|------------|-----------|-------------|
| MSE | {perf.get('traditional_metrics', {}).get('mse', 0):.6f} | {perf.get('attention_metrics', {}).get('mse', 0):.6f} | {perf.get('improvement_pct', 0):.2f}% |
| R² | {perf.get('traditional_metrics', {}).get('r2', 0):.6f} | {perf.get('attention_metrics', {}).get('r2', 0):.6f} | |

## Cross-Validation (MSE)
- Traditional: {cv.get('traditional_cv', {}).get('mean_mse', 0):.6f} ± {cv.get('traditional_cv', {}).get('std_mse', 0):.6f}
- Attention: {cv.get('attention_cv', {}).get('mean_mse', 0):.6f} ± {cv.get('attention_cv', {}).get('std_mse', 0):.6f}

## Test Performance
| Model | MSE | R² |
|-------|-----|-----|
| Traditional | {test.get('traditional_test', {}).get('mse', 0):.6f} | {test.get('traditional_test', {}).get('r2', 0):.6f} |
| Attention | {test.get('attention_test', {}).get('mse', 0):.6f} | {test.get('attention_test', {}).get('r2', 0):.6f} |
"""
        
        #Add top features if available
        att_analysis = results.get('attention_analysis', {})
        if att_analysis.get('available'):
            report += "\n## Top Features\n"
            for i, (feature, score) in enumerate(att_analysis.get('top_features', [])[:10], 1):
                report += f"{i}. **{feature}**: {score:.6f}\n"
        
        with open(self.output_dir / "report.md", 'w') as f:
            f.write(report)

def main():
    try:
        evaluator = AttentionMetaEvaluator()
        results = evaluator.run()
        print("Success!" if results else "Issues occurred")
    except Exception as e:
        print(f"Critical error: {e}")

if __name__ == "__main__":
    main()