#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xgboost as xgb
import mysql.connector
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from typing import Dict, List, Optional, Tuple
import pickle
import json
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class KernelCodeDatabase:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.connection = None
        self._load_config()
    
    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        self.mysql_config = config['mysql_config']
        self.mysql_config['database'] = 'kernel_patch2'
    
    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.mysql_config)
            print("✅ Connected to MySQL")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def load_training_data(self, limit: Optional[int] = None, batch_size: int = 10000) -> pd.DataFrame:
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        print(f"Loading {limit or 'all'} patches...")
        
        base_query = """
        SELECT 
            p.patchwork_id,
            p.subject,
            p.author_email,
            p.send_time,
            CASE WHEN c.patchwork_id IS NOT NULL THEN 1 ELSE 0 END as accepted
        FROM patch p
        LEFT JOIN commit_to_patch c ON p.patchwork_id = c.patchwork_id
        WHERE p.subject IS NOT NULL 
            AND p.subject REGEXP 'sched|scheduler|mm|memory|lock|mutex|atomic|fs|net'
            AND LENGTH(p.subject) > 10
        """
        
        if limit:
            base_query += f" LIMIT {limit}"
        
        chunks = []
        for chunk in pd.read_sql(base_query, self.connection, chunksize=batch_size):
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        df['patch_content'] = ''
        
        print(f"Loaded {len(df):,} patches, acceptance rate: {df['accepted'].mean():.2%}")
        return df
    
    def close(self):
        if self.connection:
            self.connection.close()

class HierarchicalFeatureExtractor:
    def __init__(self):
        self.subsystem_keywords = {
            'scheduler': ['sched', 'fair', 'rt', 'rq', 'task_struct', 'preempt', 'numa'],
            'memory': ['mm', 'memory', 'slab', 'kmalloc', 'page', 'alloc', 'gfp', 'leak'],
            'locking': ['lock', 'mutex', 'spin', 'atomic', 'rcu', 'deadlock', 'race'],
            'filesystem': ['fs', 'file', 'inode', 'vfs', 'mount', 'read', 'write'],
            'network': ['net', 'socket', 'tcp', 'packet', 'route', 'proto']
        }
        
        self.defaults = {
            'scheduler': ('rq', 'schedule', '!rq', 'return', 'kernel/sched/core.c'),
            'memory': ('ptr', 'kmalloc', '!ptr', 'return NULL', 'mm/slab.c'),
            'locking': ('lock', 'spin_lock', '!lock', 'return -EINVAL', 'kernel/locking/spinlock.c')
        }
    
    def extract_batch_features(self, df: pd.DataFrame, batch_size: int = 5000) -> pd.DataFrame:
        print(f"Extracting features from {len(df):,} patches...")
        features = [self._extract_single_features(row) for _, row in df.iterrows()]
        features_df = pd.DataFrame(features)
        print(f"Extracted {len(features_df.columns)} features")
        return features_df
    
    def _extract_single_features(self, row: pd.Series) -> Dict:
        subject = str(row.get('subject', '')).lower()
        author_email = str(row.get('author_email', ''))
        
        #Basic + subsystem features
        features = {
            'subject_length': len(subject), 'word_count': len(subject.split()),
            'has_colon': ':' in subject, 'has_brackets': any(c in subject for c in '[](){}'),
            'exclamation_count': subject.count('!'), 'question_count': subject.count('?'),
            'author_is_maintainer': any(d in author_email.lower() for d in ['kernel.org', 'linux.org', 'redhat.com', 'intel.com']),
            'author_domain_length': len(author_email.split('@')[-1]) if '@' in author_email else 0,
        }
        
        #Subsystem scoring
        subsystem_scores = {}
        for subsystem, keywords in self.subsystem_keywords.items():
            score = sum(1 for kw in keywords if kw in subject)
            features[f'{subsystem}_score'] = score
            features[f'has_{subsystem}'] = score > 0
            subsystem_scores[subsystem] = score
        
        #Target extraction
        primary_subsystem = max(subsystem_scores, key=subsystem_scores.get) if any(subsystem_scores.values()) else 'scheduler'
        var, func, cond, action, path = self.defaults.get(primary_subsystem, self.defaults['scheduler'])
        
        features.update({
            'target_subsystem': primary_subsystem, 'target_primary_variable': var,
            'target_function': func, 'target_condition': cond, 'target_action': action, 'target_file_path': path,
            'target_complexity': 'simple' if any(w in subject for w in ['fix', 'bug']) and len(subject.split()) <= 8 
                               else 'complex' if any(w in subject for w in ['optimize', 'improve']) or len(subject.split()) > 12 else 'medium'
        })
        
        #Temporal features
        if 'send_time' in row and pd.notna(row['send_time']):
            try:
                ts = pd.to_datetime(row['send_time'], unit='ms', errors='coerce')
                if pd.notna(ts):
                    features.update({'hour': ts.hour, 'day_of_week': ts.dayofweek, 'month': ts.month,
                                   'is_weekend': ts.dayofweek >= 5, 'is_business_hours': 9 <= ts.hour <= 17})
            except: pass
        
        return features

class HierarchicalXGBoostGenerator:
    def _build_code_templates(self) -> Dict[str, str]:
        return {
            'simple': '''--- a/{path}
+++ b/{path}
@@ -123,6 +123,10 @@ {function_signature}(void)
{{
    {type} *{primary_var}{init};

    /* Fix: {description} */
    if (unlikely({condition})) {{
        {action};
    }}

    return{ret_val};
}}''',
            'medium': '''--- a/{path}
+++ b/{path}
@@ -123,8 +123,15 @@ {function_signature}(void)
{{
    {type} *{primary_var}{init};
    {secondary_decl}

    /* Fix: {description} */
    if (unlikely({condition})) {{
        goto out;
    }}

    {additional_logic}

out:
    {action};
    return{ret_val};
}}''',
            'complex': '''--- a/{path}
+++ b/{path}
@@ -123,12 +123,25 @@ {function_signature}(void)
{{
    {type} *{primary_var}{init};
    {secondary_decl}
    unsigned long flags;

    /* Fix: {description} */
    {lock_acquire}
    
    if (unlikely({condition})) {{
        {action};
        goto unlock;
    }}

    {additional_logic}

unlock:
    {lock_release}
    return{ret_val};
}}'''
        }
    
    def __init__(self, weight_decay: float = 0.01):
        self.weight_decay = weight_decay
        self.feature_extractor = HierarchicalFeatureExtractor()
        self.database = None
        self.models = {}  #Flat structure: 'subsystem_classifier', 'scheduler_variable', etc.
        self.encoders = {}
        self.scalers = {}
        self.training_metrics = None
        self.code_templates = self._build_code_templates()
        
        #Subsystem configs for template generation
        self.subsystem_configs = {
            'scheduler': {'path': 'kernel/sched/core.c', 'type': 'struct rq', 'init': ' = this_rq()', 
                         'secondary_decl': 'struct task_struct *curr = rq->curr;', 'ret_val': '',
                         'lock_acquire': 'raw_spin_lock_irqsave(&rq->lock, flags);', 
                         'lock_release': 'raw_spin_unlock_irqrestore(&rq->lock, flags);'},
            'memory': {'path': 'mm/slab.c', 'type': 'void', 'init': '', 
                      'secondary_decl': 'struct zone *zone;', 'ret_val': ' ptr',
                      'lock_acquire': '', 'lock_release': ''},
            'locking': {'path': 'kernel/locking/spinlock.c', 'type': 'spinlock_t', 'init': '', 
                       'secondary_decl': 'struct task_struct *owner;', 'ret_val': '',
                       'lock_acquire': 'preempt_disable();', 'lock_release': 'preempt_enable();'}
        }
    
    def initialize_database(self, config_path: str = "config.json") -> bool:
        try:
            self.database = KernelCodeDatabase(config_path)
            return self.database.connect()
        except Exception as e:
            print(f"Database initialization failed: {e}")
            return False
    
    def train_hierarchical_models(self, limit: Optional[int] = None, cv_folds: int = 3) -> dict:
        if not self.database:
            raise RuntimeError("Database not initialized")
        
        print("Training Hierarchical XGBoost Models")
        start_time = time.time()
        
        df = self.database.load_training_data(limit=limit)
        features_df = self.feature_extractor.extract_batch_features(df, batch_size=5000)
        
        feature_columns = [col for col in features_df.columns if not col.startswith('target_')]
        X = features_df[feature_columns].fillna(0)
        
        self.scalers['primary'] = StandardScaler()
        X_scaled = self.scalers['primary'].fit_transform(X)
        
        model_accuracies = {}
        cv_scores = {}
        
        print("Training Level 1: Primary Classifiers")
        
        #Train primary classifiers
        for target in ['subsystem', 'complexity']:
            target_col = f'target_{target}'
            if target_col in features_df.columns:
                y = features_df[target_col]
                acc, cv = self._train_model(f'{target}_classifier', X_scaled, y, cv_folds)
                model_accuracies[target] = acc
                cv_scores[target] = cv
        
        print("Training Level 2: Subsystem-Specific Models")
        
        #Debug subsystem distribution
        subsystem_counts = features_df['target_subsystem'].value_counts()
        print(f"Subsystem distribution: {subsystem_counts.to_dict()}")
        
        #Train subsystem-specific models
        for subsystem in ['scheduler', 'memory', 'locking']:
            mask = features_df['target_subsystem'] == subsystem
            print(f"Training {subsystem} models with {mask.sum()} samples...")
            
            if mask.sum() < 50:
                print(f"Skipping {subsystem} - insufficient data ({mask.sum()} samples)")
                continue
            
            X_sub = X_scaled[mask]
            
            for pred_type in ['variable', 'function', 'condition', 'action']:
                target_col = f'target_{pred_type}' if pred_type != 'variable' else 'target_primary_variable'
                
                if target_col in features_df.columns:
                    y = features_df.loc[mask, target_col]
                    if len(y.unique()) > 1:
                        model_name = f'{subsystem}_{pred_type}'
                        acc, cv = self._train_model(model_name, X_sub, y, cv_folds)
                        model_accuracies[model_name] = acc
                        cv_scores[model_name] = cv
                        print(f"Trained {model_name}: {acc:.3f}")
                    else:
                        print(f"Skipping {subsystem}_{pred_type} - only one unique value")
        
        training_time = time.time() - start_time
        
        self.training_metrics = {
            'model_accuracies': model_accuracies,
            'cross_val_scores': cv_scores,
            'training_time': training_time
        }
        
        self._display_training_results()
        return self.training_metrics
    
    def _train_model(self, model_name: str, X: np.ndarray, y: pd.Series, cv_folds: int) -> Tuple[float, List[float]]:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y.astype(str))
        self.encoders[model_name] = encoder
        
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            reg_alpha=self.weight_decay, reg_lambda=self.weight_decay * 2,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', n_jobs=-1, tree_method='hist'
        )
        
        cv_scores = []
        if len(X) > 1000:
            kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            for train_idx, val_idx in kfold.split(X, y_encoded):
                model_fold = model
                model_fold.fit(X[train_idx], y_encoded[train_idx])
                cv_scores.append(accuracy_score(y_encoded[val_idx], model_fold.predict(X[val_idx])))
        
        model.fit(X, y_encoded)
        accuracy = accuracy_score(y_encoded, model.predict(X))
        self.models[model_name] = model
        
        return accuracy, cv_scores
    
    def generate_code_batch(self, issues: List[str], author_infos: Optional[List[Dict]] = None, batch_size: int = 100) -> List[dict]:
        print(f"Generating code for {len(issues)} issues...")
        
        if author_infos is None:
            author_infos = [{}] * len(issues)
        
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.generate_single_code, issue, author_info)
                      for issue, author_info in zip(issues, author_infos)]
            
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({
                        'code': f"/* Error: {str(e)} */", 'confidence_score': 0.0, 'accuracy_estimate': 0.0,
                        'subsystem': 'unknown', 'complexity': 'simple', 'predictions': {}, 
                        'generation_time': 0.0, 'fallback_used': True
                    })
        return results
    
    def generate_single_code(self, issue_description: str, author_info: Dict = None) -> dict:
        start_time = time.time()
        
        features = self._extract_generation_features(issue_description, author_info or {})
        features_df = pd.DataFrame([features])
        
        feature_columns = [col for col in features_df.columns if not col.startswith('target_')]
        X = features_df[feature_columns].fillna(0)
        
        training_features = self.scalers['primary'].feature_names_in_
        X_aligned = X.reindex(columns=training_features, fill_value=0)
        X_scaled = self.scalers['primary'].transform(X_aligned)
        
        predictions = {}
        confidence_scores = []
        
        #Level 1 predictions
        for target in ['subsystem', 'complexity']:
            model_name = f'{target}_classifier'
            if model_name in self.models:
                pred = self.models[model_name].predict(X_scaled)[0]
                proba = self.models[model_name].predict_proba(X_scaled)[0]
                decoded = self.encoders[model_name].inverse_transform([pred])[0]
                predictions[target] = decoded
                confidence_scores.append(np.max(proba))
            else:
                predictions[target] = 'scheduler' if target == 'subsystem' else 'simple'
                confidence_scores.append(0.5)
        
        subsystem = predictions['subsystem']
        complexity = predictions['complexity']
        
        #Level 2 predictions
        for pred_type in ['variable', 'function', 'condition', 'action']:
            model_name = f'{subsystem}_{pred_type}'
            
            if model_name in self.models:
                try:
                    pred = self.models[model_name].predict(X_scaled)[0]
                    proba = self.models[model_name].predict_proba(X_scaled)[0]
                    decoded = self.encoders[model_name].inverse_transform([pred])[0]
                    predictions[pred_type] = decoded
                    confidence_scores.append(np.max(proba))
                except Exception:
                    predictions[pred_type] = self._get_default_prediction(subsystem, pred_type)
                    confidence_scores.append(0.3)
            else:
                predictions[pred_type] = self._get_default_prediction(subsystem, pred_type)
                confidence_scores.append(0.3)
        
        generated_code = self._assemble_code(predictions, issue_description, subsystem, complexity)
        overall_confidence = np.mean(confidence_scores)
        accuracy_estimate = self._estimate_accuracy(confidence_scores)
        
        return {
            'code': generated_code, 'confidence_score': overall_confidence, 'accuracy_estimate': accuracy_estimate,
            'subsystem': subsystem, 'complexity': complexity, 'predictions': predictions, 
            'generation_time': time.time() - start_time, 'fallback_used': False
        }
    
    def _extract_generation_features(self, issue_description: str, author_info: Dict) -> Dict:
        mock_row = pd.Series({
            'subject': issue_description,
            'author_email': author_info.get('email', 'user@example.com'),
            'patch_content': '', 'send_time': None
        })
        return self.feature_extractor._extract_single_features(mock_row)
    
    def _get_default_prediction(self, subsystem: str, pred_type: str) -> str:
        defaults = {
            'scheduler': {'variable': 'rq', 'function': 'schedule', 'condition': '!rq', 'action': 'return'},
            'memory': {'variable': 'ptr', 'function': 'kmalloc', 'condition': '!ptr', 'action': 'return NULL'},
            'locking': {'variable': 'lock', 'function': 'spin_lock', 'condition': '!lock', 'action': 'return -EINVAL'}
        }
        return defaults.get(subsystem, defaults['scheduler']).get(pred_type, 'unknown')
    
    def _assemble_code(self, predictions: Dict, issue_description: str, subsystem: str, complexity: str) -> str:
        template = self.code_templates.get(complexity, self.code_templates['simple'])
        config = self.subsystem_configs.get(subsystem, self.subsystem_configs['scheduler'])
        
        template_vars = {
            'path': config['path'],
            'function_signature': predictions.get('function', 'schedule'),
            'type': config['type'], 
            'primary_var': predictions.get('variable', 'rq'),
            'init': config['init'], 
            'secondary_decl': config['secondary_decl'],
            'description': issue_description[:60] + '...' if len(issue_description) > 60 else issue_description,
            'condition': predictions.get('condition', '!rq'),
            'action': predictions.get('action', 'return'), 
            'ret_val': config['ret_val'],
            'additional_logic': self._get_additional_logic(subsystem, complexity),
            'lock_acquire': config['lock_acquire'], 
            'lock_release': config['lock_release']
        }
        
        try:
            return template.format(**template_vars)
        except KeyError:
            return f'''--- a/kernel/sched/core.c
+++ b/kernel/sched/core.c
@@ -123,6 +123,10 @@ schedule(void)
{{
    struct rq *rq = this_rq();
    /* Fix: {issue_description[:60]}... */
    if (!rq) {{ return; }}
    return;
}}'''
    
    def _get_additional_logic(self, subsystem: str, complexity: str) -> str:
        if complexity == 'simple': return ''
        logic_map = {
            'scheduler': {'medium': 'update_rq_clock(rq);', 'complex': 'update_rq_clock(rq);\n    if (rq->nr_running > 1) set_tsk_need_resched(curr);'},
            'memory': {'medium': 'memset(ptr, 0, size);', 'complex': 'if (gfp & __GFP_ZERO) memset(ptr, 0, size);'},
            'locking': {'medium': 'lockdep_assert_held(&lock->rlock);', 'complex': 'lockdep_assert_held(&lock->rlock);\n    if (lock->owner != current) lock->owner = current;'}
        }
        return logic_map.get(subsystem, {}).get(complexity, '')
    
    def _estimate_accuracy(self, confidence_scores: List[float]) -> float:
        if not self.training_metrics:
            return np.mean(confidence_scores)
        
        base_accuracy = np.mean(list(self.training_metrics['model_accuracies'].values())) if self.training_metrics['model_accuracies'] else 0.7
        confidence_adjustment = (np.mean(confidence_scores) - 0.5) * 0.3
        return max(0.0, min(1.0, base_accuracy + confidence_adjustment))
    
    def _display_training_results(self):
        if not self.training_metrics:
            return
        
        print("Training Complete!")
        
        for model_name, accuracy in self.training_metrics['model_accuracies'].items():
            cv_scores = self.training_metrics['cross_val_scores'].get(model_name, [])
            cv_display = f"{np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}" if cv_scores else "N/A"
            status = "Success" if accuracy > 0.8 else "Fine" if accuracy > 0.6 else "Failed"
            print(f"{status} {model_name}: {accuracy:.3f} (CV: {cv_display})")
        
        accuracies = list(self.training_metrics['model_accuracies'].values())
        good_models = sum(1 for acc in accuracies if acc > 0.8)
        print(f"📊 Summary: {good_models}/{len(accuracies)} good models, avg: {np.mean(accuracies):.3f}, time: {self.training_metrics['training_time']:.1f}s")
    
    def save_models(self, output_dir: Path):
        output_dir.mkdir(exist_ok=True)
        
        #Save all models and encoders
        for name, model in self.models.items():
            if model:
                with open(output_dir / f"{name}.pkl", 'wb') as f:
                    pickle.dump(model, f)
        
        with open(output_dir / "encoders.pkl", 'wb') as f:
            pickle.dump(self.encoders, f)
        with open(output_dir / "scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        if self.training_metrics:
            with open(output_dir / "training_metrics.json", 'w') as f:
                json.dump(self.training_metrics, f, indent=2, default=str)
        
        with open(output_dir / "config.json", 'w') as f:
            json.dump({'version': '1.0', 'created': datetime.now().isoformat()}, f, indent=2)
        
        print(f"💾 Saved {len(self.models)} models to {output_dir}")
    
    def load_models(self, model_dir: Path):
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        print(f"Loading models from {model_dir}...")
        
        with open(model_dir / "encoders.pkl", 'rb') as f:
            self.encoders = pickle.load(f)
        with open(model_dir / "scalers.pkl", 'rb') as f:
            self.scalers = pickle.load(f)
        
        #Load all model files
        for model_file in model_dir.glob("*.pkl"):
            if model_file.name not in ['encoders.pkl', 'scalers.pkl']:
                model_name = model_file.stem
                with open(model_file, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
        
        print(f"Loaded {len(self.models)} models")


def demonstrate_production_generator():
    """Demo output"""
    print("Production Kernel Code Generator Demo")
    
    generator = HierarchicalXGBoostGenerator(weight_decay=0.015)
    
    if not generator.initialize_database():
        print("Could not connect to database")
        return None
    
    try:
        print("Training on 50k patches...")
        generator.train_hierarchical_models(limit=50000, cv_folds=3)
        
        test_issue = "CFS scheduler shows high latency during context switches under heavy load"
        result = generator.generate_single_code(test_issue, {'email': 'expert@kernel.org'})
        
        print(f"\nGeneration Results: confidence={result['confidence_score']:.3f}, accuracy={result['accuracy_estimate']:.3f}, subsystem={result['subsystem']}, complexity={result['complexity']}, time={result['generation_time']:.3f}s")
        print(f"\nGenerated Code:\n{result['code']}")
        
        test_issues = [
            "Memory leak detected in NUMA memory allocator during stress testing",
            "Race condition in RT scheduler causes system hang during load balancing", 
            "Spinlock contention in interrupt handling path causes performance regression",
            "Page allocator shows poor performance with large memory allocations",
            "Scheduler load balancing fails to handle NUMA topology correctly"
        ]
        
        batch_results = generator.generate_code_batch(test_issues)
        
        print("\nBatch Results:")
        for i, (issue, result) in enumerate(zip(test_issues, batch_results)):
            print(f"{i+1}. {issue[:40]}... -> {result['subsystem']} (conf:{result['confidence_score']:.3f})")
        
        output_dir = Path(f"./production_kernel_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        generator.save_models(output_dir)
        
        print(f"Demo complete! Models saved to: {output_dir}")
        return generator
        
    except Exception as e:
        print(f"Demo failed: {e}")
        return None
    finally:
        if generator.database:
            generator.database.close()


def main():
    print("Production Kernel Code Generator - Training Mode")
    
    generator = HierarchicalXGBoostGenerator(weight_decay=0.015)
    
    if not generator.initialize_database():
        print("Database connection failed")
        return None
    
    try:
        print("Training models on all available data...")
        generator.train_hierarchical_models(limit=None)
        output_dir = Path(f"./production_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        generator.save_models(output_dir)
        print(f"Training complete! Models saved to: {output_dir}")
        
        #Code generation demo
        print("\nTesting Code Generation...")
        test_issues = [
            "CFS scheduler shows high latency during context switches under heavy load",
            "Memory leak detected in NUMA memory allocator during stress testing",
            "Race condition in RT scheduler causes system hang during load balancing",
            "Spinlock contention in interrupt handling path causes performance regression"
        ]
        
        for i, issue in enumerate(test_issues, 1):
            print(f"\n--- Test {i}: {issue[:50]}... ---")
            result = generator.generate_single_code(issue, {'email': 'developer@kernel.org'})
            print(f"Subsystem: {result['subsystem']}, Complexity: {result['complexity']}")
            print(f"Confidence: {result['confidence_score']:.3f}, Accuracy: {result['accuracy_estimate']:.3f}")
            print(f"Generated Code:\n{result['code']}\n")
        
        print("Code generation demo complete!")
        return generator
        
    finally:
        generator.database.close()


if __name__ == "__main__":
    main()