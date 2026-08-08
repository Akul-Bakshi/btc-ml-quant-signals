import pandas as pd
import numpy as np
import pathlib
import json
import yaml
import joblib
import sys
from collections import defaultdict
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from strategy.signals import get_combined_signal

def main():
    base_dir = pathlib.Path(__file__).parent.parent
    
    # Load Data
    print("Loading data...")
    df_15m = pd.read_csv(base_dir / "data" / "processed" / "BTC_USDT_15m_features.csv", index_col='timestamp', parse_dates=True)
    df_1h = pd.read_csv(base_dir / "data" / "processed" / "BTC_USDT_1h_features.csv", index_col='timestamp', parse_dates=True)
    df_4h = pd.read_csv(base_dir / "data" / "processed" / "BTC_USDT_4h_features.csv", index_col='timestamp', parse_dates=True)
    
    # Load Model
    print("Loading model...")
    model = joblib.load(base_dir / "models" / "saved" / "scalper_ensemble.pkl")
    with open(base_dir / "models" / "saved" / "scalper_selected_features.json", "r") as f:
        features = json.load(f)
    
    balance = 10000.0
    position = None
    trades = []
    equity_curve = []
    regime_counts = defaultdict(int)
    
    config_path = base_dir / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    cutoff_date = config.get('trading', {}).get('train_cutoff_date')
    
    start_idx = 200
    if cutoff_date:
        cutoff_ts = pd.to_datetime(cutoff_date, utc=True)
        # Find index where timestamp >= cutoff_date
        idxs = df_15m.index >= cutoff_ts
        if idxs.any():
            start_idx = max(200, np.argmax(idxs))
    
    print(f"Starting backtest on {len(df_15m)} 15m candles from index {start_idx} ({df_15m.index[start_idx]})...")
    for i in range(start_idx, len(df_15m)):
        current_time = df_15m.index[i]
        
        idx_4h = df_4h.index.searchsorted(current_time, side='right')
        df_4h_w = df_4h.iloc[max(0, idx_4h-60):idx_4h]
        
        idx_1h = df_1h.index.searchsorted(current_time, side='right')
        df_1h_w = df_1h.iloc[max(0, idx_1h-60):idx_1h]
        
        df_15m_w = df_15m.iloc[max(0, i-60):i+1]
        
        if len(df_4h_w) < 20 or len(df_1h_w) < 20: 
            continue
        
        current = df_15m.iloc[i]
        
        # POSITION MANAGEMENT
        if position is not None:
            held = i - position['entry_i']
            if position['direction'] == 'LONG':
                stop_hit = current['low'] <= position['stop_price']
                target_hit = current['high'] >= position['target_price']
            else:
                stop_hit = current['high'] >= position['stop_price']
                target_hit = current['low'] <= position['target_price']
            time_stop = held >= 6
            
            exit_reason, exit_price = None, None
            if stop_hit:
                exit_reason, exit_price = 'STOP', position['stop_price']
            elif target_hit:
                exit_reason, exit_price = 'TARGET', position['target_price']
            elif time_stop:
                exit_reason, exit_price = 'TIME', current['close']
                
            if exit_reason:
                if position['direction'] == 'LONG':
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                else:
                    pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
                
                pnl_dollars = position['position_size'] * pnl_pct
                fee = position['position_size'] * 0.0002
                net = pnl_dollars - fee
                balance += net
                
                trades.append({
                    'entry_time': str(position['entry_time']),
                    'exit_time': str(current_time),
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl_dollars': pnl_dollars,
                    'net_dollars': net,
                    'regime': position['regime'],
                    'exit_reason': exit_reason,
                    'ml_confidence': position.get('ml_confidence', 0.0)
                })
                
                if len(trades) <= 20:
                    print(f"TRADE {len(trades)} | {position['regime']} | "
                          f"{exit_reason} | {position['direction']} | "
                          f"PnL ${net:.2f} ({pnl_pct*100:.2f}%) | "
                          f"Risk {position.get('actual_risk_pct', 0)*100:.2f}% | "
                          f"Bal ${balance:.2f}")
                position = None
                
        # Update equity curve every candle (floating)
        if position is not None:
            if position['direction'] == 'LONG':
                unreal = (current['close'] - position['entry_price']) / position['entry_price']
            else:
                unreal = (position['entry_price'] - current['close']) / position['entry_price']
            eq = balance + position['position_size'] * unreal
        else:
            eq = balance
        equity_curve.append({'timestamp': str(current_time), 'equity': float(eq)})
            
        # ENTRY
        if position is not None: continue
        
        sig = get_combined_signal(df_15m_w, df_1h_w, df_4h_w, model, features)
        regime_counts[sig.get('regime', 'NONE')] += 1
        
        if sig['signal'] == 'NO_TRADE': continue
        
        if i + 1 >= len(df_15m): 
            continue
            
        entry_price = df_15m.iloc[i+1]['open']
        entry_time = df_15m.index[i+1]
        
        d = sig['signal']
        if d == 'LONG':
            stop_price = entry_price * (1 - sig['stop_pct'])
            target_price = entry_price * (1 + sig['target_pct'])
        else:
            stop_price = entry_price * (1 + sig['stop_pct'])
            target_price = entry_price * (1 - sig['target_pct'])
            
        max_position = balance * 0.95
        target_risk_pct = 0.02
        risk_amount = balance * target_risk_pct
        ideal_position = risk_amount / sig['stop_pct']
        position_size = min(ideal_position, max_position)
        actual_risk_pct = (position_size * sig['stop_pct']) / balance
        
        balance -= position_size * 0.0002  # entry fee
        
        position = {
            'entry_i': i + 1, 'entry_time': entry_time,
            'entry_price': entry_price, 'stop_price': stop_price,
            'target_price': target_price,
            'position_size': position_size, 'direction': d,
            'regime': sig['regime'],
            'actual_risk_pct': actual_risk_pct,
            'ml_confidence': sig.get('confidence', 0.0)
        }
        
    print("\nBacktest complete. Calculating metrics...")
    
    eq_series = pd.Series([x['equity'] for x in equity_curve])
    peak = eq_series.cummax()
    dd = ((eq_series - peak) / peak).min()
    wins = [t for t in trades if t['pnl_dollars'] > 0]
    
    df_eq = pd.DataFrame(equity_curve)
    if not df_eq.empty:
        df_eq['timestamp'] = pd.to_datetime(df_eq['timestamp'])
        df_eq.set_index('timestamp', inplace=True)
        daily_ret = df_eq['equity'].resample('D').last().pct_change().dropna()
        if len(daily_ret) > 0 and daily_ret.std() != 0:
            sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(365))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0
    
    win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
    total_ret = (balance - 10000.0) / 10000.0
    
    metrics = {
        'total_trades': len(trades),
        'total_return_pct': float(total_ret * 100),
        'win_rate_pct': float(win_rate * 100),
        'max_drawdown_pct': float(dd * 100),
        'sharpe_ratio': sharpe,
        'final_balance': float(balance),
        'regime_counts_in_candles': dict(regime_counts),
        'trades_per_regime': {k: int(v) for k, v in pd.Series([t['regime'] for t in trades]).value_counts().items()} if trades else {}
    }
    
    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))
    
    save_path = base_dir / "models" / "saved" / "backtest_metrics.json"
    save_dir = save_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    with open(save_dir / "trade_log.json", "w") as f:
        json.dump(trades, f)
        
    with open(save_dir / "equity_curve.json", "w") as f:
        json.dump(equity_curve, f)
        
    df_plot = df_15m.iloc[start_idx:][['open', 'high', 'low', 'close']].reset_index()
    df_plot['timestamp'] = df_plot['timestamp'].astype(str)
    with open(save_dir / "price_series.json", "w") as f:
        json.dump(df_plot.to_dict(orient='records'), f)

if __name__ == "__main__":
    main()
