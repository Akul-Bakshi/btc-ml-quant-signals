import os
import sys
import json
import yaml
import joblib
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import seaborn as sns

def get_base_dir():
    return pathlib.Path(__file__).parent.parent

def load_config():
    with open(get_base_dir() / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data():
    """Loads features and returns the last 20% test split."""
    config = load_config()
    tf = config['timeframes'].get('tertiary', '1h')
    symbol = config['symbols']['primary'].replace("/", "_")
    file_path = get_base_dir() / "data" / "processed" / f"{symbol}_{tf}_features.csv"
    
    if not file_path.exists():
        print("No processed data found. Run: python data/features.py first")
        sys.exit(0)
    
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    
    # Drop lookahead columns
    lookahead_cols = [c for c in df.columns if 'future' in c.lower() or 'forward' in c.lower() or c == 'target_return']
    df = df.drop(columns=lookahead_cols, errors='ignore').iloc[:-3]
    
    y = df['target_direction']
    X = df.drop(columns=['target_direction'])
    
    # Select features
    try:
        with open(get_base_dir() / "models" / "saved" / "selected_features.json", "r") as f:
            selected = json.load(f)
        X = X[selected]
    except:
        pass
        
    X = X.fillna(0)
    
    split_idx = int(len(X) * 0.8)
    return X.iloc[split_idx:], y.iloc[split_idx:], df['close'].iloc[split_idx:]

def evaluate_model(model_name):
    """Loads a model, predicts on test set, prints classification report."""
    print(f"\n--- Evaluating {model_name} ---")
    model_path = get_base_dir() / "models" / "saved" / f"{model_name}.pkl"
    if not model_path.exists():
        print(f"Model {model_name} not found.")
        return None, None
        
    try:
        model = joblib.load(model_path)
        X_test, y_test, _ = load_data()
        
        preds = model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, preds))
        
        cm = confusion_matrix(y_test, preds)
        print("Confusion Matrix:")
        print(cm)
        
        dir_acc = accuracy_score(y_test, preds)
        print(f"Directional Accuracy: {dir_acc:.2%}")
        
        return model, preds
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return None, None

def plot_equity_curve(predictions, actual_returns_series):
    """Simulates trading strategy against buy and hold, saves plot."""
    try:
        config = load_config()
        # forward returns actual percentage
        # since we have 'close', actual return is change in close
        close_prices = actual_returns_series
        returns = close_prices.pct_change().shift(-1).fillna(0)
        
        strategy_returns = returns * predictions
        
        cum_bh = (1 + returns).cumprod()
        cum_strat = (1 + strategy_returns).cumprod()
        
        plt.figure(figsize=(10, 6))
        plt.plot(cum_bh.index, cum_bh, label='Buy and Hold')
        plt.plot(cum_strat.index, cum_strat, label='ML Strategy')
        plt.title('Equity Curve: ML Strategy vs B&H')
        plt.ylabel('Cumulative Return Multiplier')
        plt.legend()
        
        save_path = get_base_dir() / "models" / "saved" / "equity_curve.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved equity curve to {save_path}")
    except Exception as e:
        print(f"Error plotting equity curve: {e}")

def plot_feature_importance():
    """Plots top 30 features as horizontal bar chart."""
    try:
        fi_path = get_base_dir() / "models" / "saved" / "feature_importance.json"
        with open(fi_path, "r") as f:
            fi = json.load(f)
            
        series = pd.Series(fi).sort_values(ascending=True).tail(30)
        plt.figure(figsize=(10, 8))
        series.plot(kind='barh')
        plt.title('Top 30 Feature Importances (XGB/LGBM Avg)')
        plt.xlabel('Importance')
        
        save_path = get_base_dir() / "models" / "saved" / "feature_importance.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved feature importances to {save_path}")
    except Exception as e:
        print(f"Error plotting feature importance: {e}")

def plot_confusion_matrix(model_name="ensemble_model"):
    """Saves visual confusion matrix."""
    try:
        X_test, y_test, _ = load_data()
        model_path = get_base_dir() / "models" / "saved" / f"{model_name}.pkl"
        model = joblib.load(model_path)
        preds = model.predict(X_test)
        
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({model_name})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        save_path = get_base_dir() / "models" / "saved" / "confusion_matrix.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved confusion matrix to {save_path}")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def calibration_check(model, X_test, y_test, model_name="ensemble_model"):
    """Checks and plots probability calibration."""
    try:
        if not hasattr(model, 'predict_proba'):
            print("Model does not support predict_proba, skipping calibration.")
            return
            
        probs = model.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
        plt.title(f'Calibration Curve ({model_name})')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.legend()
        
        save_path = get_base_dir() / "models" / "saved" / "calibration_curve.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved calibration curve to {save_path}")
        
        # Simple heuristic for poor calibration checking (MSE of curve)
        mse = np.mean((prob_true - prob_pred)**2)
        if mse > 0.05:
            print("Model appears poorly calibrated. Applying CalibratedClassifierCV wrapper...")
            calibrated = CalibratedClassifierCV(model, cv='prefit')
            calibrated.fit(X_test, y_test)
            joblib.dump(calibrated, get_base_dir() / "models" / "saved" / f"{model_name}_calibrated.pkl")
            print("Saved re-calibrated model.")
    except Exception as e:
        print(f"Error checking calibration: {e}")

if __name__ == "__main__":
    print("--- STEP 3c: EVALUATING MODELS ---")
    models_to_eval = ["ensemble_model", "XGBoost_model", "RandomForest_model", "LightGBM_model"]
    
    ensemble_preds = None
    close_prices = None
    
    X_test, y_test, close_p = load_data()
    close_prices = close_p
    
    for m in models_to_eval:
        mod, p = evaluate_model(m)
        if m == "ensemble_model" and p is not None:
            ensemble_preds = p
            if mod:
                calibration_check(mod, X_test, y_test, m)
                plot_confusion_matrix(m)
                
    if ensemble_preds is not None:
        plot_equity_curve(ensemble_preds, close_prices)
        
    plot_feature_importance()
