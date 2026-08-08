import os
import yaml
import pandas as pd

def get_config_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

def load_config():
    with open(get_config_path(), 'r') as f:
        return yaml.safe_load(f)

def load_raw(symbol, timeframe):
    """
    Loads raw CSV data and sets timestamp as UTC datetime index.
    """
    safe_symbol = symbol.replace('/', '_')
    file_path = os.path.join(os.path.dirname(__file__), 'raw', f"{safe_symbol}_{timeframe}.csv")
    
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def clean(df, timeframe):
    """
    Cleans raw OHLCV data. Removes anomalies, forward fills short gaps,
    and flags longer gaps. Ensures index is sorted and numeric types.
    """
    if df.empty:
        return df
        
    initial_len = len(df)
    
    # Ensure float64
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype('float64')
    
    # Remove obvious anomalies
    df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
    df = df[df['volume'] >= 0]
    df = df[df['high'] >= df['low']]
    df = df[abs(df['close'] - df['open']) / df['open'] <= 0.5]
    
    print(f"Removed {initial_len - len(df)} anomalous rows.")
    
    # Sort index and drop duplicates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    
    # Forward fill gaps of 3 candles or fewer
    tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '1d': '1D'}
    pd_tf = tf_map.get(timeframe, timeframe)
    
    try:
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=pd_tf)
        missing = full_idx.difference(df.index)
        
        if len(missing) > 0:
            df = df.reindex(full_idx)
            is_nan = df['close'].isna()
            gap_groups = is_nan.ne(is_nan.shift()).cumsum()
            gap_sizes = is_nan.groupby(gap_groups).sum()
            long_gaps = gap_sizes[gap_sizes > 3]
            
            if len(long_gaps) > 0:
                print(f"WARNING: Found {len(long_gaps)} gaps longer than 3 candles for tf {timeframe}.")
            
            df = df.ffill(limit=3)
            df = df.dropna()
    except Exception as e:
        print(f"Error during gap filling: {e}")
        
    return df

def align_timeframes(symbol):
    """
    Loads and cleans all defined timeframes for a symbol.
    Returns dictionary mapping timeframe to cleaned DataFrame.
    """
    config = load_config()
    timeframes = list(config['timeframes'].values())
    aligned_data = {}
    
    for tf in timeframes:
        print(f"Aligning {symbol} {tf}...")
        df_raw = load_raw(symbol, tf)
        if not df_raw.empty:
            df_clean = clean(df_raw, tf)
            aligned_data[tf] = df_clean
            
    return aligned_data

def save_clean(df, symbol, timeframe):
    """
    Saves cleaned dataframe to raw directory with _clean suffix.
    """
    if df.empty:
        return
        
    safe_symbol = symbol.replace('/', '_')
    raw_dir = os.path.join(os.path.dirname(__file__), 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    file_path = os.path.join(raw_dir, f"{safe_symbol}_{timeframe}_clean.csv")
    
    try:
        df.index.name = 'timestamp'
        df.reset_index().to_csv(file_path, index=False)
        print(f"Saved cleaned data to {file_path}")
    except Exception as e:
        print(f"Error saving clean data: {e}")

def clean_all():
    """ Runs cleaning process for all symbols and timeframes. """
    config = load_config()
    symbols = [config['symbols']['primary'], config['symbols']['secondary']]
    timeframes = list(config['timeframes'].values())
    
    print("Starting data clean pipeline...")
    for symbol in symbols:
        for tf in timeframes:
            print(f"\nProcessing {symbol} {tf}")
            df_raw = load_raw(symbol, tf)
            if not df_raw.empty:
                df_clean = clean(df_raw, tf)
                save_clean(df_clean, symbol, tf)
            else:
                print(f"No raw data found for {symbol} {tf}")

if __name__ == "__main__":
    clean_all()
