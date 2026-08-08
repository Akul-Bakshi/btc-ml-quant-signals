import sys
import os
import json
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ccxt
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.features import (add_trend_indicators, add_momentum_indicators, 
                           add_volatility_indicators, add_volume_indicators, 
                           add_fibonacci_features, add_golden_ratio_multiplier,
                           add_statistical_features)
from strategy.signals import get_combined_signal

def fetch_live_data(symbol='BTC/USDT', limit=500):
    exchange = ccxt.binance()
    timeframes = {'15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d'}
    dataframes = {}
    
    for name, tf in timeframes.items():
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        dataframes[name] = df
        
    return dataframes

def process_features(dfs):
    # Process 1d
    df_1d = dfs['1d'].copy()
    df_1d = add_trend_indicators(df_1d)
    df_1d = add_golden_ratio_multiplier(df_1d)
    if 'ema_200' in df_1d.columns:
        df_1d['trend_1d'] = np.where(df_1d['close'] > df_1d['ema_200'], 1, 0)
    cols_1d = [c for c in df_1d.columns if 'golden_ratio' in c] + ['ema_200', 'trend_1d']
    df_1d_sub = df_1d[[c for c in cols_1d if c in df_1d.columns]].copy().add_suffix('_1d')
    df_1d_sub.index = df_1d_sub.index + pd.Timedelta(days=1)
    
    # Process 4h
    df_4h = dfs['4h'].copy()
    df_4h = add_trend_indicators(df_4h)
    df_4h = add_momentum_indicators(df_4h)
    df_4h = add_volatility_indicators(df_4h)
    df_4h = add_fibonacci_features(df_4h)
    if 'ema_200' in df_4h.columns:
        df_4h['trend_4h'] = np.where(df_4h['close'] > df_4h['ema_200'], 1, 0)
    fib_cols = [c for c in df_4h.columns if 'fib_' in c] + ['rsi_14', 'ema_200', 'trend_4h', 'adx_14']
    df_4h_sub = df_4h[[c for c in fib_cols if c in df_4h.columns]].copy().add_suffix('_4h')
    df_4h_sub.index = df_4h_sub.index + pd.Timedelta(hours=4)
    
    # Process 1h
    df_1h = dfs['1h'].copy()
    df_1h = add_trend_indicators(df_1h)
    df_1h = add_momentum_indicators(df_1h)
    df_1h = add_volatility_indicators(df_1h)
    cols_1h = ['rsi_14', 'ema_50', 'macd_hist', 'bb_pct']
    df_1h_sub = df_1h[[c for c in cols_1h if c in df_1h.columns]].copy().add_suffix('_1h')
    df_1h_sub.index = df_1h_sub.index + pd.Timedelta(hours=1)
    
    # Process 15m
    df_15m = dfs['15m'].copy()
    import ta
    df_15m['ema_21'] = ta.trend.ema_indicator(df_15m['close'], window=21)
    df_15m['rsi_7'] = ta.momentum.rsi(df_15m['close'], window=7)
    df_15m['rsi_14'] = ta.momentum.rsi(df_15m['close'], window=14)
    stoch = ta.momentum.StochasticOscillator(df_15m['high'], df_15m['low'], df_15m['close'], window=5, smooth_window=3)
    df_15m['stoch_k'] = stoch.stoch()
    df_15m['stoch_d'] = stoch.stoch_signal()
    for p in [3, 6, 12, 24]:
        df_15m[f'roc_{p}'] = ta.momentum.roc(df_15m['close'], window=p)
    bb_10 = ta.volatility.BollingerBands(df_15m['close'], window=10, window_dev=2)
    df_15m['bb_10_upper'] = bb_10.bollinger_hband()
    df_15m['bb_10_lower'] = bb_10.bollinger_lband()
    bb_20 = ta.volatility.BollingerBands(df_15m['close'], window=20, window_dev=2)
    df_15m['bb_20_upper'] = bb_20.bollinger_hband()
    df_15m['bb_20_lower'] = bb_20.bollinger_lband()
    df_15m['atr_7'] = ta.volatility.average_true_range(df_15m['high'], df_15m['low'], df_15m['close'], window=7)
    df_15m['atr_14'] = ta.volatility.average_true_range(df_15m['high'], df_15m['low'], df_15m['close'], window=14)
    
    roll_mean_10 = df_15m['close'].rolling(window=10).mean()
    roll_std_10 = df_15m['close'].rolling(window=10).std()
    df_15m['zscore_10'] = (df_15m['close'] - roll_mean_10) / roll_std_10
    
    roll_mean_20 = df_15m['close'].rolling(window=20).mean()
    roll_std_20 = df_15m['close'].rolling(window=20).std()
    df_15m['zscore_20'] = (df_15m['close'] - roll_mean_20) / roll_std_20
    
    hl_range = df_15m['high'] - df_15m['low']
    hl_range = hl_range.replace(0, np.nan)
    df_15m['volume_delta'] = ((df_15m['close'] - df_15m['low']) / hl_range) * df_15m['volume'] - df_15m['volume'] / 2
    
    vol_mean = df_15m['volume'].rolling(window=10).mean()
    vol_std = df_15m['volume'].rolling(window=10).std()
    df_15m['volume_zscore_10'] = (df_15m['volume'] - vol_mean) / vol_std
    
    vol_mean_20 = df_15m['volume'].rolling(window=20).mean()
    vol_std_20 = df_15m['volume'].rolling(window=20).std()
    df_15m['volume_zscore_20'] = (df_15m['volume'] - vol_mean_20) / vol_std_20
    
    df_15m['obv'] = ta.volume.on_balance_volume(df_15m['close'], df_15m['volume'])
    
    # Merge
    df = pd.merge_asof(df_15m.sort_index(), df_4h_sub.sort_index(), left_index=True, right_index=True, direction='backward')
    df = pd.merge_asof(df.sort_index(), df_1h_sub.sort_index(), left_index=True, right_index=True, direction='backward')
    df = pd.merge_asof(df.sort_index(), df_1d_sub.sort_index(), left_index=True, right_index=True, direction='backward')
    
    return df, df_1h, df_4h

def load_ml_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(base, "models", "saved", "scalper_ensemble.pkl"))
    with open(os.path.join(base, "models", "saved", "scalper_selected_features.json"), "r") as f:
        features = json.load(f)
    return model, features

def update_live_stats(signals_df):
    stats = {
        'total_signals': len(signals_df),
        'open_count': len(signals_df[signals_df['status'] == 'OPEN']),
        'wins': len(signals_df[signals_df['status'] == 'WIN']),
        'losses': len(signals_df[signals_df['status'] == 'LOSS']),
        'time_exits': len(signals_df[signals_df['status'] == 'TIME']),
        'live_win_rate': 0.0,
        'avg_win_pct': 0.0,
        'avg_loss_pct': 0.0,
        'cumulative_return_pct': 0.0
    }
    
    closed = signals_df[signals_df['status'].isin(['WIN', 'LOSS', 'TIME'])]
    if len(closed) > 0:
        stats['live_win_rate'] = (stats['wins'] / len(closed)) * 100
        stats['cumulative_return_pct'] = closed['pnl_pct'].sum() * 100
        
        wins_df = closed[closed['pnl_pct'] > 0]
        loss_df = closed[closed['pnl_pct'] <= 0]
        if len(wins_df) > 0: stats['avg_win_pct'] = wins_df['pnl_pct'].mean() * 100
        if len(loss_df) > 0: stats['avg_loss_pct'] = loss_df['pnl_pct'].mean() * 100
        
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, 'live_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    return stats

def run_tracker():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching live data...")
    dfs = fetch_live_data()
    df_15m_feat, df_1h, df_4h = process_features(dfs)
    
    model, features = load_ml_model()
    
    # We want to use the last fully CLOSED 15m candle for signaling
    # The last row in ccxt is usually the forming candle. 
    # We'll use iloc[-2] as the last closed candle.
    closed_15m = df_15m_feat.iloc[:-1]
    last_closed_idx = closed_15m.index[-1]
    
    # Prepare windows for signal generator
    df_15m_w = df_15m_feat.loc[:last_closed_idx].iloc[-60:]
    df_1h_w = df_1h.loc[:last_closed_idx].iloc[-60:]
    df_4h_w = df_4h.loc[:last_closed_idx].iloc[-60:]
    
    sig = get_combined_signal(df_15m_w, df_1h_w, df_4h_w, model, features)
    
    current_price = dfs['15m'].iloc[-1]['close']
    
    # Load or create tracking CSV
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, 'live_signals.csv')
    if os.path.exists(csv_path):
        signals_df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    else:
        signals_df = pd.DataFrame(columns=[
            'timestamp', 'signal', 'regime', 'ml_confidence', 
            'entry_price', 'stop_price', 'target_price', 'status', 'pnl_pct'
        ])
    
    # Update OPEN signals
    for idx, row in signals_df[signals_df['status'] == 'OPEN'].iterrows():
        # Check high/low since entry
        entry_time = row['timestamp']
        # Data from entry_time to now
        recent_data = dfs['15m'].loc[entry_time:]
        if len(recent_data) > 0:
            max_high = recent_data['high'].max()
            min_low = recent_data['low'].min()
            current_close = recent_data.iloc[-1]['close']
            
            status = 'OPEN'
            exit_price = 0
            
            if row['signal'] == 'LONG':
                if min_low <= row['stop_price']:
                    status, exit_price = 'LOSS', row['stop_price']
                elif max_high >= row['target_price']:
                    status, exit_price = 'WIN', row['target_price']
                elif len(recent_data) >= 6:
                    status, exit_price = 'TIME', current_close
            else:
                if max_high >= row['stop_price']:
                    status, exit_price = 'LOSS', row['stop_price']
                elif min_low <= row['target_price']:
                    status, exit_price = 'WIN', row['target_price']
                elif len(recent_data) >= 6:
                    status, exit_price = 'TIME', current_close
                    
            if status != 'OPEN':
                pnl = (exit_price - row['entry_price']) / row['entry_price']
                if row['signal'] == 'SHORT': pnl = -pnl
                signals_df.at[idx, 'status'] = status
                signals_df.at[idx, 'pnl_pct'] = pnl
                print(f"Signal from {entry_time} closed as {status} (PnL: {pnl*100:.2f}%)")

    # Add new signal if it exists and we don't already have one for this candle
    if sig['signal'] in ['LONG', 'SHORT']:
        if len(signals_df) == 0 or signals_df.iloc[-1]['timestamp'] != last_closed_idx:
            new_row = {
                'timestamp': last_closed_idx,
                'signal': sig['signal'],
                'regime': sig['regime'],
                'ml_confidence': sig.get('confidence', 0),
                'entry_price': current_price,
                'stop_price': current_price * (1 - sig['stop_pct']) if sig['signal'] == 'LONG' else current_price * (1 + sig['stop_pct']),
                'target_price': current_price * (1 + sig['target_pct']) if sig['signal'] == 'LONG' else current_price * (1 - sig['target_pct']),
                'status': 'OPEN',
                'pnl_pct': 0.0
            }
            signals_df = pd.concat([signals_df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"NEW SIGNAL! {sig['signal']} | Regime: {sig['regime']} | Conf: {sig.get('confidence', 0):.2f}")
    
    signals_df.to_csv(csv_path, index=False)
    
    stats = update_live_stats(signals_df)
    
    print("\n=== LIVE TRACKER SUMMARY ===")
    print(f"Current Regime: {sig.get('regime', 'NONE')}")
    print(f"Current Signal: {sig.get('signal', 'NONE')} (Conf: {sig.get('confidence', 0):.2f})")
    print(f"Total Tracked: {stats['total_signals']} | Open: {stats['open_count']}")
    print(f"Win Rate: {stats['live_win_rate']:.2f}% | Return: {stats['cumulative_return_pct']:.2f}%")

if __name__ == "__main__":
    run_tracker()
