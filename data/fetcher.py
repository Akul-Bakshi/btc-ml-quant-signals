import os
import time
import ccxt
import yaml
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# Load env variables (assumes .env is in the root directory)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def get_config_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

def load_config():
    with open(get_config_path(), 'r') as f:
        return yaml.safe_load(f)

def fetch_ohlcv(symbol, timeframe, lookback_days):
    """
    Connects to Binance and fetches OHLCV data going back lookback_days.
    Paginated to handle the 1000 candle limit.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    now = datetime.now(timezone.utc)
    since_dt = now - timedelta(days=lookback_days)
    since = int(since_dt.timestamp() * 1000)
    
    all_ohlcv = []
    print(f"Fetching {symbol} {timeframe} since {since_dt.strftime('%Y-%m-%d')}")
    
    while since < int(now.timestamp() * 1000):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            
            if len(ohlcv) < 1000:
                break
                
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df

def save_raw(df, symbol, timeframe):
    """
    Saves the fetched OHLCV dataframe to CSV in data/raw/.
    Appends new rows and drops duplicates if file already exists.
    """
    safe_symbol = symbol.replace('/', '_')
    raw_dir = os.path.join(os.path.dirname(__file__), 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    file_path = os.path.join(raw_dir, f"{safe_symbol}_{timeframe}.csv")
    
    if os.path.exists(file_path):
        print(f"File {file_path} exists. Appending and dropping duplicates...")
        existing_df = pd.read_csv(file_path)
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], utc=True)
        
        combined_df = pd.concat([existing_df, df])
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        combined_df.to_csv(file_path, index=False)
        print(f"Saved {len(combined_df) - len(existing_df)} new rows to {file_path}")
    else:
        df.sort_values('timestamp').reset_index(drop=True).to_csv(file_path, index=False)
        print(f"Created new file with {len(df)} rows at {file_path}")

def fetch_all():
    """
    Fetches and saves data for every symbol and timeframe in config.yaml.
    """
    config = load_config()
    lookback_days = config['data']['lookback_days']
    symbols = [config['symbols']['primary'], config['symbols']['secondary']]
    timeframes = list(config['timeframes'].values())
    
    print("Starting data fetch pipeline...")
    for symbol in symbols:
        for tf in timeframes:
            try:
                df = fetch_ohlcv(symbol, tf, lookback_days)
                if not df.empty:
                    save_raw(df, symbol, tf)
                else:
                    print(f"No data returned for {symbol} {tf}")
            except Exception as e:
                print(f"Failed to process {symbol} {tf}: {e}")

if __name__ == "__main__":
    fetch_all()
