#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from datetime import datetime

#ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import lightgbm as lgb

#Try transformers
TRANSFORMERS_AVAILABLE = False
try:
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
    import torch
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = True
    print("Transformers loaded")
except:
    print("Transformers unavailable")

class TfidfTextClassifier:
    def __init__(self, classifier_type='logistic', max_features=10000, ngram_range=(1, 2)):
        self.classifier_type = classifier_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.pipeline = None
        
    def fit(self, X, y):
        clf = LogisticRegression(random_state=42, max_iter=1000) if self.classifier_type == 'logistic' else MultinomialNB()
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english',
                token_pattern=r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',
                min_df=2,
                max_df=0.95
            )),
            ('classifier', clf)
        ])
        X_clean = [str(x) if pd.notna(x) and str(x).strip() else "unknown" for x in X]
        self.pipeline.fit(X_clean, y)
        return self
    
    def predict(self, X):
        X_clean = [str(x) if pd.notna(x) and str(x).strip() else "unknown" for x in X]
        return self.pipeline.predict(X_clean)
    
    def predict_proba(self, X):
        X_clean = [str(x) if pd.notna(x) and str(x).strip() else "unknown" for x in X]
        return self.pipeline.predict_proba(X_clean)

if TRANSFORMERS_AVAILABLE:
    class CommitDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx]).strip() if str(self.texts[idx]).strip() else "unknown"
            encoding = self.tokenizer(text, truncation=True, padding='max_length', 
                                     max_length=self.max_length, return_tensors='pt')
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

    class TransformerWrapper:
        def __init__(self, model_name='distilbert-base-uncased', max_length=128):
            self.model_name = model_name
            self.max_length = max_length
            self.tokenizer = None
            self.model = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        def fit(self, X, y):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2).to(self.device)
            
            dataset = CommitDataset(X, y, self.tokenizer, self.max_length)
            
            args = TrainingArguments(
                output_dir='./temp_trainer',
                num_train_epochs=3,
                per_device_train_batch_size=8,
                warmup_steps=100,
                logging_steps=50,
                save_strategy="no",
                evaluation_strategy="no"
            )
            
            trainer = Trainer(model=self.model, args=args, train_dataset=dataset, tokenizer=self.tokenizer)
            trainer.train()
            return self
        
        def predict(self, X):
            clf = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer,
                          device=0 if torch.cuda.is_available() else -1, truncation=True, max_length=self.max_length)
            texts = [str(x).strip() if str(x).strip() else "unknown" for x in X]
            results = clf(texts)
            return np.array([1 if r['label'] == 'LABEL_1' else 0 for r in results])
        
        def predict_proba(self, X):
            clf = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer,
                          device=0 if torch.cuda.is_available() else -1, return_all_scores=True,
                          truncation=True, max_length=self.max_length)
            texts = [str(x).strip() if str(x).strip() else "unknown" for x in X]
            results = clf(texts)
            probs = []
            for r in results:
                if isinstance(r, list):
                    probs.append([r[0]['score'], r[1]['score']])
                else:
                    probs.append([0.5, 0.5])
            return np.array(probs)

class ModelDevelopment:
    def __init__(self):
        self.output_dir = Path(f"./phase2_primary_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(exist_ok=True)
        self.best_models = {}
        self.evaluation_results = {}
        self.test_results = {}
        
    def load_data(self):
        #Find data files
        paths = [Path("./phase1_output"), Path("./phase1_output/phase2_ml_data")]
        feature_files = []
        target_files = []
        
        for p in paths:
            if p.exists():
                feature_files.extend(list(p.glob("X_features_*.csv")))
                target_files.extend(list(p.glob("y_target_*.csv")))
        
        if not feature_files:
            raise FileNotFoundError("No Phase 1 data found")
        
        self.X = pd.read_csv(max(feature_files, key=lambda x: x.stat().st_mtime))
        self.y = pd.read_csv(max(target_files, key=lambda x: x.stat().st_mtime)).iloc[:, 0]
        self.X = self.X.fillna(0)
        
        #Find text column
        self.text_column = None
        for col in ['subject', 'commit_subject', 'commit_message', 'text']:
            if col in self.X.columns:
                texts = self.X[col].dropna().head(10).astype(str)
                if texts.str.len().mean() > 20 and len(texts.unique())/len(texts) > 0.7:
                    self.text_column = col
                    break
        
        if not self.text_column:
            for col in self.X.columns:
                if self.X[col].dtype == 'object':
                    self.text_column = col
                    break
        
        self.feature_names = [c for c in self.X.columns if c != self.text_column]
        self.X_numerical = self.X[self.feature_names]
        
        print(f"Loaded {len(self.X)} samples, {len(self.feature_names)} features, text column: {self.text_column}")
        
    def split_data(self):
        #Train/val/test split
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42)
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
        
        self.X_train_num = self.X_train[self.feature_names]
        self.X_val_num = self.X_val[self.feature_names]
        self.X_test_num = self.X_test[self.feature_names]
        
        if self.text_column:
            self.X_train_text = self.X_train[self.text_column].values
            self.X_val_text = self.X_val[self.text_column].values
            self.X_test_text = self.X_test[self.text_column].values
        
    def train_models(self):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        #Model configurations
        scale_pos_weight = len(self.y_train[self.y_train==0]) / max(len(self.y_train[self.y_train==1]), 1)
        
        models = {
            'rf': (RandomForestClassifier(random_state=42, n_jobs=-1), 
                   {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'class_weight': ['balanced', None]},
                   'numerical'),
            'xgb': (xgb.XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight),
                    {'n_estimators': [100, 200], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1]},
                    'numerical'),
            'lgb': (lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced'),
                    {'n_estimators': [100, 200], 'max_depth': [3, 6, -1], 'learning_rate': [0.01, 0.1]},
                    'numerical')
        }
        
        #Add text models if text column exists
        if self.text_column:
            models['tfidf'] = (TfidfTextClassifier(), 
                              {'max_features': [5000, 10000], 'ngram_range': [(1,1), (1,2)]},
                              'text')
            
            if TRANSFORMERS_AVAILABLE:
                models['distilbert'] = (TransformerWrapper('distilbert-base-uncased'),
                                      {'max_length': [128]},
                                      'text')
        
        #Train each model
        for name, (model, params, dtype) in models.items():
            print(f"\nTraining {name}...")
            
            if dtype == 'numerical':
                X_tr, X_vl = self.X_train_num, self.X_val_num
            else:
                X_tr, X_vl = self.X_train_text, self.X_val_text
            
            try:
                if name == 'distilbert' and TRANSFORMERS_AVAILABLE:
                    #Simple transformer training
                    model.fit(X_tr, self.y_train.values)
                    self.best_models[name] = {'model': model, 'params': params, 'dtype': dtype}
                elif name == 'tfidf':
                    #Grid search for TF-IDF
                    best_score = 0
                    best_model = None
                    for mf in params['max_features']:
                        for ng in params['ngram_range']:
                            m = TfidfTextClassifier(max_features=mf, ngram_range=ng)
                            m.fit(X_tr, self.y_train)
                            score = f1_score(self.y_val, m.predict(X_vl))
                            if score > best_score:
                                best_score = score
                                best_model = m
                    self.best_models[name] = {'model': best_model, 'params': params, 'dtype': dtype}
                else:
                    #RandomizedSearchCV for traditional ML
                    search = RandomizedSearchCV(model, params, n_iter=10, cv=cv, scoring='f1', 
                                              n_jobs=-1, random_state=42, verbose=0)
                    search.fit(X_tr, self.y_train)
                    self.best_models[name] = {'model': search.best_estimator_, 
                                             'params': search.best_params_, 'dtype': dtype}
                
                #Evaluate
                preds = self.best_models[name]['model'].predict(X_vl)
                f1 = f1_score(self.y_val, preds)
                print(f"{name} F1: {f1:.4f}")
                
                try:
                    proba = self.best_models[name]['model'].predict_proba(X_vl)
                    if len(proba.shape) > 1:
                        proba = proba[:, 1]
                    auc = roc_auc_score(self.y_val, proba)
                except:
                    auc = 0.5
                
                self.evaluation_results[name] = {
                    'f1': f1, 'auc': auc,
                    'precision': precision_score(self.y_val, preds, zero_division=0),
                    'recall': recall_score(self.y_val, preds, zero_division=0),
                    'accuracy': accuracy_score(self.y_val, preds)
                }
                self.best_models[name]['val_proba'] = proba
                
            except Exception as e:
                print(f"Error with {name}: {str(e)[:100]}")
                
    def test_models(self):
        print("\nTest evaluation...")
        
        for name, model_data in self.best_models.items():
            if 'model' not in model_data:
                continue
                
            model = model_data['model']
            dtype = model_data['dtype']
            
            X_test = self.X_test_num if dtype == 'numerical' else self.X_test_text
            
            preds = model.predict(X_test)
            f1 = f1_score(self.y_test, preds)
            
            try:
                proba = model.predict_proba(X_test)
                if len(proba.shape) > 1:
                    proba = proba[:, 1]
                auc = roc_auc_score(self.y_test, proba)
            except:
                auc = 0.5
                proba = preds.astype(float)
            
            self.test_results[name] = {
                'f1': f1, 'auc': auc,
                'precision': precision_score(self.y_test, preds, zero_division=0),
                'recall': recall_score(self.y_test, preds, zero_division=0),
                'predictions': proba
            }
            print(f"{name} Test F1: {f1:.4f}")
    
    def create_ensemble(self):
        #Weighted ensemble
        weights = {}
        predictions = {}
        
        for name, model_data in self.best_models.items():
            if 'val_proba' in model_data and name in self.evaluation_results:
                predictions[name] = model_data['val_proba']
                weights[name] = self.evaluation_results[name]['f1']
        
        if len(predictions) < 2:
            return None
        
        total_weight = sum(weights.values()) or 1
        ensemble_proba = sum(weights[n]/total_weight * predictions[n] for n in predictions)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        print(f"\nEnsemble F1: {f1_score(self.y_val, ensemble_pred):.4f}")
        
        #Test ensemble
        if self.test_results:
            test_preds = {}
            for name in predictions.keys():
                if name in self.test_results:
                    test_preds[name] = self.test_results[name]['predictions']
            
            if test_preds:
                test_ensemble = sum(weights[n]/total_weight * test_preds[n] for n in test_preds)
                test_pred = (test_ensemble > 0.5).astype(int)
                print(f"Ensemble Test F1: {f1_score(self.y_test, test_pred):.4f}")
    
    def save_results(self):
        #Save models
        models_dir = self.output_dir / "trained_models"
        models_dir.mkdir(exist_ok=True)
        
        for name, model_data in self.best_models.items():
            if 'model' in model_data:
                try:
                    with open(models_dir / f"{name}_kernel_model.pkl", 'wb') as f:
                        pickle.dump(model_data['model'], f)
                except:
                    pass
        
        #Save results JSON
        results = {
            'validation_results': self.evaluation_results,
            'test_results': {k: {m: v for m, v in r.items() if m != 'predictions'} 
                           for k, r in self.test_results.items()},
            'best_hyperparameters': {k: v.get('params', {}) for k, v in self.best_models.items()},
            'dataset_info': {
                'text_column': self.text_column,
                'total_samples': len(self.X),
                'feature_count': len(self.feature_names),
                'acceptance_rate': float(self.y.mean())
            }
        }
        
        with open(self.output_dir / "kernel_evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        #Save feature importance
        importance_data = {}
        for name, model_data in self.best_models.items():
            if model_data['dtype'] == 'numerical' and hasattr(model_data['model'], 'feature_importances_'):
                importance_data[name] = model_data['model'].feature_importances_
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data, index=self.feature_names)
            importance_df['ensemble_importance'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values('ensemble_importance', ascending=False)
            importance_df.to_csv(self.output_dir / "kernel_feature_importance.csv")
        
        #Save splits
        with open(self.output_dir / "kernel_data_splits.json", 'w') as f:
            json.dump({
                'train_indices': self.X_train.index.tolist(),
                'val_indices': self.X_val.index.tolist(),
                'test_indices': self.X_test.index.tolist(),
                'text_column': self.text_column,
                'feature_names': self.feature_names
            }, f, indent=2)
        
        #Create report
        best_model = max(self.evaluation_results.items(), key=lambda x: x[1]['f1'])
        report = f"""# Kernel Commit Scheduling Analysis

## Dataset
- Samples: {len(self.X):,}
- Features: {len(self.feature_names)}
- Text column: {self.text_column}

## Best Model
- {best_model[0]}: F1={best_model[1]['f1']:.4f}

## All Results
"""
        for name, metrics in self.evaluation_results.items():
            test_f1 = self.test_results.get(name, {}).get('f1', 'N/A')
            report += f"- {name}: Val F1={metrics['f1']:.4f}, Test F1={test_f1}\n"
        
        report += f"\nGenerated: {datetime.now()}"
        
        with open(self.output_dir / "kernel_summary_report.md", 'w') as f:
            f.write(report)
        
        print(f"\nResults saved to {self.output_dir}")

def main():
    dev = ModelDevelopment()
    dev.load_data()
    dev.split_data()
    dev.train_models()
    dev.test_models()
    dev.create_ensemble()
    dev.save_results()
    return dev

if __name__ == "__main__":
    main()