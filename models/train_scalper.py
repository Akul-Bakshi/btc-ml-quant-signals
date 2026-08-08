import sys
import json
import yaml
import pathlib
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

def get_base_dir():
    return pathlib.Path(__file__).parent.parent

def load_config():
    config_path = get_base_dir() / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_features(symbol, timeframe):
    safe_symbol = symbol.replace("/", "_")
    file_path = get_base_dir() / "data" / "processed" / f"{safe_symbol}_{timeframe}_features.csv"
    
    if not file_path.exists():
        print(f"No processed data found at {file_path}. Run: python data/features.py first")
        sys.exit(0)
        
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        print(f"Loaded {symbol} {timeframe}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading features: {e}")
        sys.exit(1)

def prepare_data(df):
    try:
        # Filter out rows with small target returns (only train on clear moves)
        if 'target_return' in df.columns:
            df = df[df['target_return'].abs() >= 0.003]
            
        config = load_config()
        cutoff_date = config.get('trading', {}).get('train_cutoff_date')
        if cutoff_date:
            df = df.loc[df.index < pd.to_datetime(cutoff_date, utc=True)]
            
        lookahead_cols = [c for c in df.columns if 'future' in c.lower() or 'forward' in c.lower() or c == 'target_return']
        df = df.drop(columns=lookahead_cols, errors='ignore')
        
        df = df.iloc[:-3]
        
        y = df['target_direction']
        X = df.drop(columns=['target_direction'])
        
        pct_1 = (y.sum() / len(y)) * 100
        pct_0 = 100 - pct_1
        print(f"Class Balance: {pct_1:.1f}% positive (1), {pct_0:.1f}% negative (0)")
        
        return X, y
    except Exception as e:
        print(f"Error preparing data: {e}")
        return pd.DataFrame(), pd.Series()

def walk_forward_split(X, y, n_splits=8):
    try:
        fold_size = len(X) // (n_splits + 1)
        for i in range(n_splits):
            train_end = (i + 1) * fold_size
            test_end = (i + 2) * fold_size
            
            X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
            X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
            
            X_train = X_train + np.random.normal(0, 0.001, X_train.shape)
            
            yield X_train, X_test, y_train, y_test, i+1
    except Exception as e:
        print(f"Error generating walk-forward splits: {e}")

def select_features(X, y):
    print("\nRunning Feature Selection (Mutual Information)...")
    try:
        mi_scores = mutual_info_classif(X.fillna(0), y)
        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        top_25 = mi_series.head(25).index.tolist()
        
        print("Top 20 Features by Mutual Info:")
        print(mi_series.head(20))
        
        save_dir = get_base_dir() / "models" / "saved"
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "scalper_selected_features.json", "w") as f:
            json.dump(top_25, f)
            
        return X[top_25], top_25
    except Exception as e:
        print(f"Error selecting features: {e}")
        return X, X.columns.tolist()

def check_overfitting(train_scores, test_scores, dir_accs):
    try:
        mean_train = np.mean(train_scores)
        mean_test = np.mean(test_scores)
        if mean_train > mean_test + 0.08:
            print(f"STRONG OVERFIT WARNING: Mean Train Acc {mean_train:.2f} vs Test Acc {mean_test:.2f}")
            
        if len(test_scores) > 1 and test_scores[-1] < test_scores[0] - 0.05:
            print(f"REGIME SHIFT WARNING: Model degraded from {test_scores[0]:.2f} (first) to {test_scores[-1]:.2f} (last)")
            
    except Exception as e:
        print(f"Error in overfitting checks: {e}")

def get_models():
    models = {
        'lgbm': LGBMClassifier(n_estimators=300, max_depth=6, num_leaves=6, min_child_samples=100, reg_lambda=5.0, reg_alpha=1.0, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=5, learning_rate=0.05, class_weight='balanced', verbosity=-1)
    }
    return models

def build_ensemble(fitted_models, weights, X_train, y_train):
    try:
        estimators = [(name, mod) for name, mod in fitted_models.items()]
        voting_clf = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
        voting_clf.fit(X_train, y_train)
        return voting_clf
    except Exception as e:
        print(f"Error building ensemble: {e}")
        return None

def save_models(fitted_models, ensemble_model, top_features, metrics_dict):
    try:
        save_dir = get_base_dir() / "models" / "saved"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for name, mod in fitted_models.items():
            joblib.dump(mod, save_dir / f"scalper_{name}.pkl")
            
        if ensemble_model:
            joblib.dump(ensemble_model, save_dir / "scalper_ensemble.pkl")
            
        with open(save_dir / "scalper_training_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
            
        config = load_config()
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": [config['symbols']['primary'], config['symbols']['secondary']],
            "timeframe": '15m',
            "n_features": len(top_features),
            "walk_forward_splits": 8
        }
        with open(save_dir / "scalper_training_metadata.json", "w") as f:
            json.dump(meta, f, indent=4)
        print("\nAll models and metadata securely saved.")
    except Exception as e:
        print(f"Error saving models: {e}")

def main():
    print("--- SCALPER MODEL TRAINING PIPELINE ---")
    config = load_config()
    tf = '15m'
    symbol = config['symbols']['primary']
    n_splits = 8
    
    df = load_features(symbol, tf)
    X, y = prepare_data(df)
    
    X, top_features = select_features(X, y)
    X = X.fillna(0)
    
    models = get_models()
    
    pos_cases = y.sum()
    neg_cases = len(y) - pos_cases
    # models['xgboost'].set_params(scale_pos_weight=neg_cases/pos_cases if pos_cases > 0 else 1)
    
    metrics_summary = {name: {'train_acc': [], 'test_acc': [], 'dir_acc': [], 'auc': []} for name in models.keys()}
    final_fitted_models = {}
    
    for X_train, X_test, y_train, y_test, fold in walk_forward_split(X, y, n_splits=n_splits):
        for name, model in models.items():
            model.fit(X_train, y_train)
            
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            test_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else test_preds
            
            tr_acc = accuracy_score(y_train, train_preds)
            te_acc = accuracy_score(y_test, test_preds)
            dir_acc = accuracy_score(y_test, test_preds)
            
            try: auc = roc_auc_score(y_test, test_probs)
            except: auc = 0.5
            
            metrics_summary[name]['train_acc'].append(tr_acc)
            metrics_summary[name]['test_acc'].append(te_acc)
            metrics_summary[name]['dir_acc'].append(dir_acc)
            metrics_summary[name]['auc'].append(auc)
            
            if fold == n_splits:
                final_fitted_models[name] = model

    weights = []
    print("\n--- FINAL METRICS (Across Folds) ---")
    for name, m_dict in metrics_summary.items():
        avg_te = np.mean(m_dict['test_acc'])
        avg_dir = np.mean(m_dict['dir_acc'])
        avg_auc = np.mean(m_dict['auc'])
        print(f"{name}:\n  Acc: {avg_te:.2f} | DirAcc: {avg_dir:.2f} | AUC: {avg_auc:.2f}")
        check_overfitting(m_dict['train_acc'], m_dict['test_acc'], m_dict['dir_acc'])
        weights.append(avg_auc)
    
    xgb_mod = final_fitted_models.get('xgboost')
    lgb_mod = final_fitted_models.get('lgbm')
    
    fi_dict = {}
    if xgb_mod and hasattr(xgb_mod, 'feature_importances_'):
        for i, val in enumerate(xgb_mod.feature_importances_):
            col = top_features[i]
            fi_dict[col] = fi_dict.get(col, 0) + val * 0.5
            
    if lgb_mod and hasattr(lgb_mod, 'feature_importances_'):
        l_fi = lgb_mod.feature_importances_
        if l_fi.sum() > 0: l_fi = l_fi / l_fi.sum()
        for i, val in enumerate(l_fi):
            col = top_features[i]
            fi_dict[col] = fi_dict.get(col, 0) + val * 0.5
            
    fi_series = pd.Series(fi_dict).sort_values(ascending=False)
    print("\nTop 20 Features (Averaged XGB/LGBM):")
    print(fi_series.head(20))
    save_dir = get_base_dir() / "models" / "saved"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "scalper_feature_importance.json", "w") as f:
        json.dump(fi_dict, f)

    # Load other models to rebuild ensemble
    final_fitted_models['logreg'] = joblib.load(save_dir / "scalper_logreg.pkl")
    final_fitted_models['rf'] = joblib.load(save_dir / "scalper_rf.pkl")
    final_fitted_models['xgboost'] = joblib.load(save_dir / "scalper_xgboost.pkl")
    
    with open(save_dir / "scalper_training_metrics.json", "r") as f:
        old_metrics = json.load(f)
        
    old_metrics['lgbm'] = metrics_summary['lgbm']
    weights = []
    for name in final_fitted_models.keys():
        weights.append(np.mean(old_metrics[name]['auc']))

    vote = build_ensemble(final_fitted_models, weights, X, y)
    save_models(final_fitted_models, vote, top_features, old_metrics)

if __name__ == "__main__":
    main()
