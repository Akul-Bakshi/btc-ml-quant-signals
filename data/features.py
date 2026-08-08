import os
import yaml
import numpy as np
import pandas as pd
import ta
import scipy.stats as stats
import statsmodels.api as sm

def get_config_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

def load_config():
    with open(get_config_path(), 'r') as f:
        return yaml.safe_load(f)

def load_clean(symbol: str, timeframe: str) -> pd.DataFrame:
    """Loads cleaned CSV and returns DataFrame with datetime index."""
    safe_symbol = symbol.replace('/', '_')
    file_path = os.path.join(os.path.dirname(__file__), 'raw', f"{safe_symbol}_{timeframe}_clean.csv")
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading clean data {file_path}: {e}")
        return pd.DataFrame()

# --- INDICATOR FEATURES ---

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds EMA, MACD, Ichimoku, and Supertrend to DataFrame."""
    try:
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
        df['adx_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        
        macd = ta.trend.MACD(df['close'])
        df['macd_line'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_tenkan'] = ichimoku.ichimoku_conversion_line()
        df['ichimoku_kijun'] = ichimoku.ichimoku_base_line()
        df['ichimoku_senkou_a'] = ichimoku.ichimoku_a()
        df['ichimoku_senkou_b'] = ichimoku.ichimoku_b()
        
        atr_10 = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=10)
        hl2 = (df['high'] + df['low']) / 2
        df['supertrend_up'] = hl2 - (3 * atr_10)
        df['supertrend_dn'] = hl2 + (3 * atr_10)
    except Exception as e:
        print(f"Error train trend indicators: {e}")
    return df

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds RSI, Stochastic %K/%D, and ROC."""
    try:
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        for p in [5, 10, 20, 50]:
            df[f'roc_{p}'] = ta.momentum.roc(df['close'], window=p)
    except Exception as e:
        print(f"Error momentum indicators: {e}")
    return df

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Bollinger Bands, ATR, and historical volatility."""
    try:
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_pband()
        df['bb_pct'] = bb.bollinger_wband()
        df['bb_bandwidth'] = bb.bollinger_wband()
        
        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        log_ret = np.log(df['close'] / df['close'].shift(1))
        df['hist_vol_20'] = log_ret.rolling(window=20).std() * np.sqrt(365)
    except Exception as e:
        print(f"Error volatility indicators: {e}")
    return df

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds VWAP, OBV, volume delta, and volume Z-score."""
    try:
        vwap = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=14)
        df['vwap'] = vwap.volume_weighted_average_price()
        
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        hl_range = df['high'] - df['low']
        hl_range = hl_range.replace(0, np.nan) 
        df['volume_delta'] = ((df['close'] - df['low']) / hl_range) * df['volume'] - df['volume'] / 2
        
        vol_mean = df['volume'].rolling(window=20).mean()
        vol_std = df['volume'].rolling(window=20).std()
        df['volume_zscore_20'] = (df['volume'] - vol_mean) / vol_std
    except Exception as e:
        print(f"Error volume indicators: {e}")
    return df

# --- MATHEMATICAL / QUANT FEATURES ---

def add_fibonacci_features(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """Finds swing high/low over rolling window, computes Fib features."""
    try:
        config = load_config()
        fib_config = config['mathematical_strategies']['fibonacci']
        ret_levels = fib_config['retracement_levels']
        ext_levels = fib_config['extension_levels']
        
        roll_high = df['high'].rolling(window=lookback).max()
        roll_low = df['low'].rolling(window=lookback).min()
        diff = roll_high - roll_low
        
        levels_to_track = []
        for level in ret_levels + ext_levels:
            val = roll_low + (diff * level)
            df[f'fib_{level}'] = val
            col_name = f'fib_{level}_pct_dist'
            df[col_name] = (df['close'] - val) / val
            levels_to_track.append(col_name)
            
        df['closest_fib_dist'] = df[levels_to_track].abs().min(axis=1)
    except Exception as e:
        print(f"Error fibonacci features: {e}")
    return df

def add_harmonic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds harmonic pattern detections."""
    try:
        df['harmonic_pattern'] = 'None'
        df['harmonic_confidence'] = 0.0
    except Exception as e:
        print(f"Error harmonic features: {e}")
    return df

def add_golden_ratio_multiplier(df: pd.DataFrame) -> pd.DataFrame:
    """Adds SMA * Golden Ratio levels."""
    try:
        sma_350 = df['close'].rolling(window=350).mean()
        multipliers = [1.0, 1.6, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]
        
        for m in multipliers:
            val = sma_350 * m
            df[f'golden_ratio_{m}_dist'] = (df['close'] - val) / val
    except Exception as e:
        print(f"Error golden ratio: {e}")
    return df

def add_pi_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Pi Cycle features: gap and crossover."""
    try:
        sma_111 = df['close'].rolling(window=111).mean()
        sma_350_x2 = df['close'].rolling(window=350).mean() * 2
        
        df['pi_cycle_gap'] = (sma_111 - sma_350_x2) / df['close']
        df['pi_cycle_cross'] = np.where(sma_111 > sma_350_x2, 1, 0)
    except Exception as e:
        print(f"Error pi cycle: {e}")
    return df

# --- STATISTICAL / QUANT FEATURES ---

def hurst_exponent(ts_data):
    """Calculates Hurst Exponent using Rescaled Range methodology."""
    try:
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts_data[lag:], ts_data[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return np.nan

def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Z-score, log returns, autocorrelation, and hurst exponent."""
    try:
        roll_mean = df['close'].rolling(window=20).mean()
        roll_std = df['close'].rolling(window=20).std()
        df['zscore_20'] = (df['close'] - roll_mean) / roll_std
        
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['autocorr_20_lag1'] = df['log_ret'].rolling(window=20).apply(lambda x: pd.Series(x).autocorr(lag=1))
        df['hurst_100'] = df['close'].rolling(window=100).apply(hurst_exponent, raw=True)
    except Exception as e:
        print(f"Error statistical features: {e}")
    return df

def add_cross_asset_features(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates cross-asset features between BTC and ETH."""
    try:
        common_idx = btc_df.index.intersection(eth_df.index)
        btc = btc_df.loc[common_idx].copy()
        eth = eth_df.loc[common_idx]
        
        btc['eth_btc_ratio'] = eth['close'] / btc['close']
        
        roll_mean = btc['eth_btc_ratio'].rolling(window=20).mean()
        roll_std = btc['eth_btc_ratio'].rolling(window=20).std()
        btc['eth_btc_zscore_20'] = (btc['eth_btc_ratio'] - roll_mean) / roll_std
        
        btc_ret = np.log(btc['close'] / btc['close'].shift(1))
        eth_ret = np.log(eth['close'] / eth['close'].shift(1))
        
        btc['btc_eth_corr_30'] = btc_ret.rolling(window=30).corr(eth_ret)
        
        mean_60 = btc['eth_btc_ratio'].rolling(window=60).mean()
        std_60 = btc['eth_btc_ratio'].rolling(window=60).std()
        btc['spread_dev_60'] = (btc['eth_btc_ratio'] - mean_60) / std_60
        
        return btc
    except Exception as e:
        print(f"Error cross-asset features: {e}")
        return btc_df

# --- MULTI-TIMEFRAME FEATURES ---

def add_mtf_features(df_base: pd.DataFrame, df_higher: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Merges higher timeframe features onto base timeframe using merge_asof."""
    try:
        cols = ['rsi_14', 'ema_50', 'ema_200', 'atr_14', 'macd_hist', 'bb_pct']
        avail_cols = [c for c in cols if c in df_higher.columns]
        
        df_h = df_higher[avail_cols].copy()
        df_h = df_h.add_suffix(suffix)
        
        merged_df = pd.merge_asof(
            df_base.sort_index(),
            df_h.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        return merged_df
    except Exception as e:
        print(f"Error adding MTF features with suffix {suffix}: {e}")
        return df_base

# --- TARGET LABEL ---

def add_target_label(df: pd.DataFrame, forward_periods: int = 3) -> pd.DataFrame:
    """Adds target based on forward periods shift."""
    try:
        future_close = df['close'].shift(-forward_periods)
        df['target_return'] = (future_close / df['close']) - 1
        df['target_direction'] = np.where(df['target_return'] > 0, 1, 0)
        df.dropna(subset=['target_return'], inplace=True)
    except Exception as e:
        print(f"Error adding target labels: {e}")
    return df

# --- PIPELINE ---

def build_features(symbols):
    """Runs all add_* functions for multiple timeframes."""
    def apply_all_features(data_df):
        if data_df.empty: return data_df
        df = data_df.copy()
        df = add_trend_indicators(df)
        df = add_momentum_indicators(df)
        df = add_volatility_indicators(df)
        df = add_volume_indicators(df)
        df = add_fibonacci_features(df, lookback=100)
        df = add_harmonic_features(df)
        df = add_golden_ratio_multiplier(df)
        df = add_pi_cycle_features(df)
        df = add_statistical_features(df)
        return df

    for sym in symbols:
        for tf in ['1d', '4h', '1h']:
            df = load_clean(sym, tf)
            if df.empty:
                print(f"Empty data for {sym} {tf}, skipping.")
                continue
            print(f"Processing {tf} features for {sym}...")
            df = apply_all_features(df)
            df.dropna(inplace=True)
            
            safe_symbol = sym.replace('/', '_')
            proc_dir = os.path.join(os.path.dirname(__file__), 'processed')
            os.makedirs(proc_dir, exist_ok=True)
            out_path = os.path.join(proc_dir, f"{safe_symbol}_{tf}_features.csv")
            df.to_csv(out_path)
            print(f"Saved {len(df)} rows to {out_path}.")


def build_15m_features(symbol: str):
    print(f"Building 15m scalper features for {symbol}...")
    df = load_clean(symbol, '15m')
    if df.empty: return
    
    # EMA 21
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    
    # Momentum
    df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=5, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    for p in [3, 6, 12, 24]:
        df[f'roc_{p}'] = ta.momentum.roc(df['close'], window=p)
        
    # Volatility
    bb_10 = ta.volatility.BollingerBands(df['close'], window=10, window_dev=2)
    df['bb_10_upper'] = bb_10.bollinger_hband()
    df['bb_10_lower'] = bb_10.bollinger_lband()
    bb_20 = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_20_upper'] = bb_20.bollinger_hband()
    df['bb_20_lower'] = bb_20.bollinger_lband()
    df['atr_7'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=7)
    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    roll_mean_10 = df['close'].rolling(window=10).mean()
    roll_std_10 = df['close'].rolling(window=10).std()
    df['zscore_10'] = (df['close'] - roll_mean_10) / roll_std_10
    
    roll_mean_20 = df['close'].rolling(window=20).mean()
    roll_std_20 = df['close'].rolling(window=20).std()
    df['zscore_20'] = (df['close'] - roll_mean_20) / roll_std_20
    
    # Volume
    hl_range = df['high'] - df['low']
    hl_range = hl_range.replace(0, np.nan)
    df['volume_delta'] = ((df['close'] - df['low']) / hl_range) * df['volume'] - df['volume'] / 2
    
    vol_mean = df['volume'].rolling(window=10).mean()
    vol_std = df['volume'].rolling(window=10).std()
    df['volume_zscore_10'] = (df['volume'] - vol_mean) / vol_std
    
    vol_mean_20 = df['volume'].rolling(window=20).mean()
    vol_std_20 = df['volume'].rolling(window=20).std()
    df['volume_zscore_20'] = (df['volume'] - vol_mean_20) / vol_std_20
    
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # MTF Context
    safe_symbol = symbol.replace('/', '_')
    base_dir = os.path.dirname(__file__)
    
    df_4h_path = os.path.join(base_dir, 'processed', f"{safe_symbol}_4h_features.csv")
    df_1h_path = os.path.join(base_dir, 'processed', f"{safe_symbol}_1h_features.csv")
    df_1d_path = os.path.join(base_dir, 'processed', f"{safe_symbol}_1d_features.csv")
    
    if os.path.exists(df_4h_path):
        df_4h = pd.read_csv(df_4h_path, index_col='timestamp', parse_dates=True)
        fib_cols = [c for c in df_4h.columns if 'fib_' in c]
        fib_cols.extend(['rsi_14', 'ema_200'])
        avail_4h = [c for c in fib_cols if c in df_4h.columns]
        df_4h_sub = df_4h[avail_4h].copy()
        if 'ema_200' in df_4h.columns:
            df_4h_sub['trend_4h'] = np.where(df_4h['close'] > df_4h['ema_200'], 1, 0)
        df_4h_sub = df_4h_sub.add_suffix('_4h')
        df_4h_sub.index = df_4h_sub.index + pd.Timedelta(hours=4)
        df = pd.merge_asof(df.sort_index(), df_4h_sub.sort_index(), left_index=True, right_index=True, direction='backward')
        
    if os.path.exists(df_1h_path):
        df_1h = pd.read_csv(df_1h_path, index_col='timestamp', parse_dates=True)
        cols_1h = ['rsi_14', 'ema_50', 'macd_hist', 'bb_pct']
        avail_1h = [c for c in cols_1h if c in df_1h.columns]
        df_1h_sub = df_1h[avail_1h].copy().add_suffix('_1h')
        df_1h_sub.index = df_1h_sub.index + pd.Timedelta(hours=1)
        df = pd.merge_asof(df.sort_index(), df_1h_sub.sort_index(), left_index=True, right_index=True, direction='backward')
        
    if os.path.exists(df_1d_path):
        df_1d = pd.read_csv(df_1d_path, index_col='timestamp', parse_dates=True)
        cols_1d = [c for c in df_1d.columns if 'golden_ratio' in c] + ['ema_200']
        avail_1d = [c for c in cols_1d if c in df_1d.columns]
        df_1d_sub = df_1d[avail_1d].copy()
        if 'ema_200' in df_1d.columns:
            df_1d_sub['trend_1d'] = np.where(df_1d['close'] > df_1d['ema_200'], 1, 0)
        df_1d_sub = df_1d_sub.add_suffix('_1d')
        df_1d_sub.index = df_1d_sub.index + pd.Timedelta(days=1)
        df = pd.merge_asof(df.sort_index(), df_1d_sub.sort_index(), left_index=True, right_index=True, direction='backward')

    # Target label
    future_close = df['close'].shift(-2)
    df['target_return'] = (future_close / df['close']) - 1
    
    df.dropna(subset=['target_return'], inplace=True)
    # df = df[df['target_return'].abs() >= 0.003] # Moved to training script
    
    df['target_direction'] = np.where(df['target_return'] > 0, 1, 0)
    df.dropna(inplace=True)
    
    out_path = os.path.join(base_dir, 'processed', f"{safe_symbol}_15m_features.csv")
    df.to_csv(out_path)
    print(f"Saved {len(df)} rows to {out_path}.")

def build_5m_features(symbol: str):
    print(f"Building 5m scalper features for {symbol}...")
    df = load_clean(symbol, '5m')
    if df.empty: return
    
    # EMA 21
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    
    # Momentum
    df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=5, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    for p in [3, 6, 12, 24]:
        df[f'roc_{p}'] = ta.momentum.roc(df['close'], window=p)
        
    # Volatility
    bb_10 = ta.volatility.BollingerBands(df['close'], window=10, window_dev=2)
    df['bb_10_upper'] = bb_10.bollinger_hband()
    df['bb_10_lower'] = bb_10.bollinger_lband()
    bb_20 = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_20_upper'] = bb_20.bollinger_hband()
    df['bb_20_lower'] = bb_20.bollinger_lband()
    df['atr_7'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=7)
    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    roll_mean_10 = df['close'].rolling(window=10).mean()
    roll_std_10 = df['close'].rolling(window=10).std()
    df['zscore_10'] = (df['close'] - roll_mean_10) / roll_std_10
    
    roll_mean_20 = df['close'].rolling(window=20).mean()
    roll_std_20 = df['close'].rolling(window=20).std()
    df['zscore_20'] = (df['close'] - roll_mean_20) / roll_std_20
    
    # Volume
    hl_range = df['high'] - df['low']
    hl_range = hl_range.replace(0, np.nan)
    df['volume_delta'] = ((df['close'] - df['low']) / hl_range) * df['volume'] - df['volume'] / 2
    
    vol_mean = df['volume'].rolling(window=10).mean()
    vol_std = df['volume'].rolling(window=10).std()
    df['volume_zscore_10'] = (df['volume'] - vol_mean) / vol_std
    
    vol_mean_20 = df['volume'].rolling(window=20).mean()
    vol_std_20 = df['volume'].rolling(window=20).std()
    df['volume_zscore_20'] = (df['volume'] - vol_mean_20) / vol_std_20
    
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # MTF Context
    safe_symbol = symbol.replace('/', '_')
    base_dir = os.path.dirname(__file__)
    
    df_4h_path = os.path.join(base_dir, 'processed', f"{safe_symbol}_4h_features.csv")
    df_1h_path = os.path.join(base_dir, 'processed', f"{safe_symbol}_1h_features.csv")
    df_1d_path = os.path.join(base_dir, 'processed', f"{safe_symbol}_1d_features.csv")
    
    if os.path.exists(df_4h_path):
        df_4h = pd.read_csv(df_4h_path, index_col='timestamp', parse_dates=True)
        fib_cols = [c for c in df_4h.columns if 'fib_' in c]
        fib_cols.extend(['rsi_14', 'ema_200'])
        avail_4h = [c for c in fib_cols if c in df_4h.columns]
        df_4h_sub = df_4h[avail_4h].copy()
        if 'ema_200' in df_4h.columns:
            df_4h_sub['trend_4h'] = np.where(df_4h['close'] > df_4h['ema_200'], 1, 0)
        df_4h_sub = df_4h_sub.add_suffix('_4h')
        df_4h_sub.index = df_4h_sub.index + pd.Timedelta(hours=4)
        df = pd.merge_asof(df.sort_index(), df_4h_sub.sort_index(), left_index=True, right_index=True, direction='backward')
        
    if os.path.exists(df_1h_path):
        df_1h = pd.read_csv(df_1h_path, index_col='timestamp', parse_dates=True)
        cols_1h = ['rsi_14', 'ema_50', 'macd_hist', 'bb_pct']
        avail_1h = [c for c in cols_1h if c in df_1h.columns]
        df_1h_sub = df_1h[avail_1h].copy().add_suffix('_1h')
        df_1h_sub.index = df_1h_sub.index + pd.Timedelta(hours=1)
        df = pd.merge_asof(df.sort_index(), df_1h_sub.sort_index(), left_index=True, right_index=True, direction='backward')
        
    if os.path.exists(df_1d_path):
        df_1d = pd.read_csv(df_1d_path, index_col='timestamp', parse_dates=True)
        cols_1d = [c for c in df_1d.columns if 'golden_ratio' in c] + ['ema_200']
        avail_1d = [c for c in cols_1d if c in df_1d.columns]
        df_1d_sub = df_1d[avail_1d].copy()
        if 'ema_200' in df_1d.columns:
            df_1d_sub['trend_1d'] = np.where(df_1d['close'] > df_1d['ema_200'], 1, 0)
        df_1d_sub = df_1d_sub.add_suffix('_1d')
        df_1d_sub.index = df_1d_sub.index + pd.Timedelta(days=1)
        df = pd.merge_asof(df.sort_index(), df_1d_sub.sort_index(), left_index=True, right_index=True, direction='backward')

    # Target label
    future_close = df['close'].shift(-2)
    df['target_return'] = (future_close / df['close']) - 1
    
    df.dropna(subset=['target_return'], inplace=True)
    # df = df[df['target_return'].abs() >= 0.003] # Moved to training script
    
    df['target_direction'] = np.where(df['target_return'] > 0, 1, 0)
    df.dropna(inplace=True)
    
    out_path = os.path.join(base_dir, 'processed', f"{safe_symbol}_5m_features.csv")
    df.to_csv(out_path)
    print(f"Saved {len(df)} rows to {out_path}.")


def build_all():
    """Builds features for configured symbols."""
    config = load_config()
    primary = config['symbols']['primary']
    secondary = config['symbols']['secondary']
    symbols = [primary, secondary]
    build_features(symbols)
    for sym in symbols:
        build_15m_features(sym)
        build_5m_features(sym)

if __name__ == "__main__":
    from fetcher import fetch_all
    from cleaner import clean_all
    
    print("--- PIPELINE START ---")
    fetch_all()
    clean_all()
    build_all()
    print("--- PIPELINE COMPLETE ---")
