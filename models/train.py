"""
# TRAINING ORDER:
# 1. python data/features.py     (build processed CSVs)
# 2. python models/train.py      (train all models, save to models/saved/)
# 3. python models/evaluate.py   (check performance, plot equity curve)
# 4. streamlit run dashboard/app.py  (view signals live)
# GA NOTE: strategy/genetic.py is wired up in Step 4 (backtest engine)
"""
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
    """Loads feature CSV, parses timestamp, and returns gracefully if missing."""
    safe_symbol = symbol.replace("/", "_")
    file_path = get_base_dir() / "data" / "processed" / f"{safe_symbol}_{timeframe}_features.csv"
    
    if not file_path.exists():
        print("No processed data found. Run: python data/features.py first")
        sys.exit(0)
        
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        print(f"Loaded {symbol} {timeframe}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Date Range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        print(f"Error loading features: {e}")
        sys.exit(1)

def prepare_data(df):
    """Drops lookahead columns, drops last incomplete rows, extracts X and y."""
    try:
        lookahead_cols = [c for c in df.columns if 'future' in c.lower() or 'forward' in c.lower() or c == 'target_return']
        df = df.drop(columns=lookahead_cols, errors='ignore')
        
        # Drop last 3 rows which might have incomplete target labels due to forward shift
        df = df.iloc[:-3]
        
        y = df['target_direction']
        X = df.drop(columns=['target_direction'])
        
        # Class balance check
        pct_1 = (y.sum() / len(y)) * 100
        pct_0 = 100 - pct_1
        print(f"Class Balance: {pct_1:.1f}% positive (1), {pct_0:.1f}% negative (0)")
        
        if pct_1 > 60 or pct_1 < 40:
            print("WARNING: Warning significant class imbalance detected > 60/40")
            
        return X, y
    except Exception as e:
        print(f"Error preparing data: {e}")
        return pd.DataFrame(), pd.Series()

def walk_forward_split(X, y, n_splits=6):
    """Yields sequential train/test splits without shuffling to avoid future leak."""
    try:
        fold_size = len(X) // (n_splits + 1)
        for i in range(n_splits):
            train_end = (i + 1) * fold_size
            test_end = (i + 2) * fold_size
            
            X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
            X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
            
            print(f"\nFold {i+1} Date Range:")
            print(f"Train: {X_train.index.min()} to {X_train.index.max()}")
            print(f"Test:  {X_test.index.min()} to {X_test.index.max()}")
            
            yield X_train, X_test, y_train, y_test, i+1
    except Exception as e:
        print(f"Error generating walk-forward splits: {e}")

def select_features(X, y):
    """Scores features using Mutual Information and returns top 50."""
    print("\nRunning Feature Selection (Mutual Information)...")
    try:
        mi_scores = mutual_info_classif(X.fillna(0), y)
        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        top_50 = mi_series.head(50).index.tolist()
        
        print("Top 20 Features by Mutual Info:")
        print(mi_series.head(20))
        
        # Save feature list
        save_dir = get_base_dir() / "models" / "saved"
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "selected_features.json", "w") as f:
            json.dump(top_50, f)
            
        return X[top_50], top_50
    except Exception as e:
        print(f"Error selecting features: {e}")
        return X, X.columns.tolist()

def check_overfitting(train_scores, test_scores, dir_accs):
    """Checks for standard signs of overfitting and regime shift."""
    try:
        mean_train = np.mean(train_scores)
        mean_test = np.mean(test_scores)
        if mean_train > mean_test + 0.10:
            print(f"⚠️ STRONG OVERFITTING WARNING: Mean Train Acc {mean_train:.2f} vs Test Acc {mean_test:.2f}")
            
        if len(test_scores) > 1 and test_scores[-1] < test_scores[0] - 0.05:
            print(f"⚠️ REGIME SHIFT WARNING: Model degraded from {test_scores[0]:.2f} (first) to {test_scores[-1]:.2f} (last)")
            
        for i, d in enumerate(dir_accs):
            if d < 0.52:
                print(f"⚠️ RANDOM FOLD WARNING: Fold {i+1} directional accuracy is {d:.2f} (<0.52)")
    except Exception as e:
        print(f"Error in overfitting checks: {e}")

def get_models():
    """Initializes strictly configured ML classifiers."""
    models = {
        'LogisticRegression': LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20, class_weight='balanced', n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, num_leaves=31, class_weight='balanced', verbosity=-1)
    }
    return models

def build_ensemble(fitted_models, weights, X_train, y_train):
    """Builds a weighted soft voting ensemble, and tries a Stacking ensemble."""
    try:
        estimators = [(name, mod) for name, mod in fitted_models.items()]
        
        voting_clf = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
        voting_clf.fit(X_train, y_train)
        
        stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(class_weight='balanced'), cv=3)
        stacking_clf.fit(X_train, y_train)
        
        return voting_clf, stacking_clf
    except Exception as e:
        print(f"Error building ensemble: {e}")
        return None, None

def save_models(fitted_models, ensemble_model, stacking_model, top_features, metrics_dict):
    """Persists model objects and meta mapping for evaluation step."""
    try:
        save_dir = get_base_dir() / "models" / "saved"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for name, mod in fitted_models.items():
            joblib.dump(mod, save_dir / f"{name}_model.pkl")
            
        if ensemble_model:
            joblib.dump(ensemble_model, save_dir / "ensemble_model.pkl")
        if stacking_model:
            joblib.dump(stacking_model, save_dir / "stacking_model.pkl")
            
        with open(save_dir / "training_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
            
        config = load_config()
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": [config['symbols']['primary'], config['symbols']['secondary']],
            "timeframe": config['timeframes']['tertiary'],
            "n_features": len(top_features),
            "walk_forward_splits": config['data']['walk_forward_windows']
        }
        with open(save_dir / "training_metadata.json", "w") as f:
            json.dump(meta, f, indent=4)
        print("\nAll models and metadata securely saved.")
    except Exception as e:
        print(f"Error saving models: {e}")

def main():
    print("--- STEP 3a: MODEL TRAINING PIPELINE ---")
    config = load_config()
    tf = config['timeframes'].get('tertiary', '1h') 
    symbol = config['symbols']['primary']
    n_splits = config['data']['walk_forward_windows']
    
    df = load_features(symbol, tf)
    X, y = prepare_data(df)
    
    X, top_features = select_features(X, y)
    X = X.fillna(0) # fallback
    
    models = get_models()
    
    # Scale pos weight for XGB based on balance dynamically
    pos_cases = y.sum()
    neg_cases = len(y) - pos_cases
    models['XGBoost'].set_params(scale_pos_weight=neg_cases/pos_cases if pos_cases > 0 else 1)
    
    metrics_summary = {name: {'train_acc': [], 'test_acc': [], 'dir_acc': [], 'auc': []} for name in models.keys()}
    
    final_fitted_models = {}
    
    for X_train, X_test, y_train, y_test, fold in walk_forward_split(X, y, n_splits=n_splits):
        actual_directions = (X_test['close'] > X_train['close'].iloc[-1]).astype(int) if 'close' in X_test.columns else y_test
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            test_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else test_preds
            
            tr_acc = accuracy_score(y_train, train_preds)
            te_acc = accuracy_score(y_test, test_preds)
            
            # directional accuracy: did we guess the binary target correctly
            dir_acc = accuracy_score(y_test, test_preds) # Assuming target is direction itself
            
            try: auc = roc_auc_score(y_test, test_probs)
            except: auc = 0.5
            
            metrics_summary[name]['train_acc'].append(tr_acc)
            metrics_summary[name]['test_acc'].append(te_acc)
            metrics_summary[name]['dir_acc'].append(dir_acc)
            metrics_summary[name]['auc'].append(auc)
            
            # Keep latest trained model
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
    
    # Extract Feature Importances for XGB and LGBM
    xgb_mod = final_fitted_models.get('XGBoost')
    lgb_mod = final_fitted_models.get('LightGBM')
    
    fi_dict = {}
    if xgb_mod and hasattr(xgb_mod, 'feature_importances_'):
        for i, val in enumerate(xgb_mod.feature_importances_):
            col = top_features[i]
            fi_dict[col] = fi_dict.get(col, 0) + val * 0.5
            
    if lgb_mod and hasattr(lgb_mod, 'feature_importances_'):
        # Normalize lgb
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
    with open(save_dir / "feature_importance.json", "w") as f:
        json.dump(fi_dict, f)

    # Ensemble full retrain on latest dataset equivalent (using the final X_train)
    vote, stack = build_ensemble(final_fitted_models, weights, X, y)
    
    save_models(final_fitted_models, vote, stack, top_features, metrics_summary)

def train_with_evolved_params(evolved_params: dict):
    """
    Called by strategy/genetic.py after parameter evolution.
    evolved_params contains evolved values for:
    - rsi_period, bb_period, zscore_lookback
    - confidence_threshold, forward_periods
    - feature_selection_k (how many features to keep)
    This function re-runs the full training pipeline 
    with these parameters substituted into config,
    returns the ensemble model and its metrics dict.
    Backtest engine (Step 4) uses the metrics as fitness score.
    """
    pass  # wired up in Step 4

if __name__ == "__main__":
    main()
