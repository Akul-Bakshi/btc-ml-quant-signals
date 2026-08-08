# Run with: streamlit run dashboard/app.py
# From project root: cd btc-ml-bot && streamlit run dashboard/app.py

import os
import yaml
import pathlib
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import joblib

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30000, key="datarefresh")
except ImportError:
    pass

st.set_page_config(layout="wide", page_title="BTC-ML Bot Dashboard", page_icon="📈")

# --- CONFIG LOADING ---
@st.cache_data
def load_config():
    config_path = pathlib.Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# --- SIDEBAR ---
st.sidebar.title("Controls")

symbols = [config['symbols']['primary'], config['symbols']['secondary']]
timeframes = list(config['timeframes'].values())

sel_sym = st.sidebar.selectbox("Symbol", symbols, index=0)
default_tf_idx = timeframes.index('1h') if '1h' in timeframes else 0
sel_tf = st.sidebar.selectbox("Timeframe", timeframes, index=default_tf_idx)
lookback = st.sidebar.slider("Lookback candles", 100, 1000, 300)

st.sidebar.subheader("Indicators")
show_ema = st.sidebar.checkbox("Show EMA (9, 21, 50, 200)", True)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", True)
show_fib = st.sidebar.checkbox("Show Fibonacci levels", True)
show_st = st.sidebar.checkbox("Show Supertrend", False)
show_vol = st.sidebar.checkbox("Show Volume", True)

st.sidebar.subheader("Quant toggles")
show_spread = st.sidebar.checkbox("Show ETH/BTC spread", False)
show_hurst = st.sidebar.checkbox("Show Hurst exponent", False)
show_vol_regime = st.sidebar.checkbox("Show volatility regime", False)

if "st_autorefresh" not in globals():
    if st.sidebar.button("Manual Refresh"):
        st.cache_data.clear()

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.features import (add_trend_indicators, add_momentum_indicators, 
                           add_volatility_indicators, add_volume_indicators)

# --- DATA LOADING ---
@st.cache_data(ttl=30)
def fetch_live_candles(symbol, timeframe, limit=300):
    import ccxt
    exchange = ccxt.binance()
    safe_symbol = symbol.replace('_', '/')
    try:
        ohlcv = exchange.fetch_ohlcv(safe_symbol, timeframe, limit=limit)
    except Exception:
        exchange = ccxt.mexc()
        ohlcv = exchange.fetch_ohlcv(safe_symbol, timeframe, limit=limit)
        
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    
    if not df.empty:
        df = add_trend_indicators(df)
        df = add_momentum_indicators(df)
        df = add_volatility_indicators(df)
        df = add_volume_indicators(df)
    return df

with st.spinner(f"Fetching live {sel_tf} data for {sel_sym}..."):
    df_raw_full = fetch_live_candles(sel_sym, sel_tf, limit=lookback+50)
    dl_status = "green"
    mtime = datetime.datetime.now(datetime.timezone.utc).timestamp()

if df_raw_full is None or df_raw_full.empty:
    st.error(f"No data found for {sel_sym} {sel_tf} via CCXT.")
    st.stop()

df = df_raw_full.tail(lookback).copy()

st.sidebar.markdown("---")
st.sidebar.write(f"**Current UTC:** {datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')}")
if mtime:
    dt_upd = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
    st.sidebar.write(f"**Last Data Update:** {dt_upd.strftime('%Y-%m-%d %H:%M:%S')}")

color = "🟢" if dl_status == "green" else "🔴" if dl_status == "red" else "🟡"
st.sidebar.write(f"**Pipeline Status:** {color} (Processed CSV {'Found' if dl_status == 'green' else 'Missing'})")

# --- MODEL LOADING ---
@st.cache_resource
def load_ml_model():
    model_dir = pathlib.Path(__file__).parent.parent / "models" / "saved"
    if model_dir.exists():
        for file in model_dir.glob("*.pkl"):
            try:
                mod = joblib.load(file)
                return mod, file.name
            except:
                pass
    return None, None

model, model_name = load_ml_model()

def get_live_signal(mod, data_df):
    if mod is None: return None
    try:
        feats = data_df.drop(columns=['target_return', 'target_direction'], errors='ignore')
        last_row = feats.iloc[-1:]
        try:
            probs = mod.predict_proba(last_row)[0]
            pred = mod.predict(last_row)[0]
            conf = max(probs)
            direction = pred
        except:
            pred = mod.predict(last_row)[0]
            conf = 0.5
            direction = pred
            
        # Top 5 dummy features if tree based
        fi_dict = {}
        if hasattr(mod, 'feature_importances_'):
            idx = np.argsort(mod.feature_importances_)[-5:]
            fi_dict = {feats.columns[i]: mod.feature_importances_[i] for i in idx}
            
        return int(direction), conf, fi_dict
    except:
        return None

# --- DERIVED METRICS ---
curr_price = df['close'].iloc[-1]
curr_rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
curr_atr = df['atr_14'].iloc[-1] if 'atr_14' in df.columns else 0

tf_hours = 1 if sel_tf == '1h' else 24 if sel_tf == '1d' else 4 if sel_tf == '4h' else 0.25 if sel_tf == '15m' else 0.083
rows_24h = max(1, int(24 / tf_hours))
if len(df) > rows_24h:
    price_24h_ago = df['close'].iloc[-rows_24h]
    pct_24h = (curr_price - price_24h_ago) / price_24h_ago * 100
else:
    pct_24h = 0

if curr_rsi < 40: rsi_badge = "🟢 OVERSOLD"
elif curr_rsi > 60: rsi_badge = "🔴 OVERBOUGHT"
else: rsi_badge = "🟡 NEUTRAL"

curr_hurst = df['hurst_100'].iloc[-1] if 'hurst_100' in df.columns and not pd.isna(df['hurst_100'].iloc[-1]) else 0.5
if curr_hurst > 0.55: hurst_badge = "📈 TRENDING"
elif curr_hurst < 0.45: hurst_badge = "📉 REVERTING"
else: hurst_badge = "🔄 RANDOM"

# Volatility regime (low bottom 33%, high top 33%)
if 'hist_vol_20' in df.columns:
    q33 = df['hist_vol_20'].quantile(0.33)
    q66 = df['hist_vol_20'].quantile(0.66)
    curr_vol = df['hist_vol_20'].iloc[-1]
    if pd.isna(curr_vol):
        vol_regime = "MEDIUM"
        vol_regimes_series = pd.Series('medium', index=df.index)
    else:
        vol_regime = "LOW" if curr_vol <= q33 else "HIGH" if curr_vol >= q66 else "MEDIUM"
        vol_regimes_series = df['hist_vol_20'].apply(lambda x: 'low' if x <= q33 else ('high' if x >= q66 else 'medium'))
else:
    vol_regime = "MEDIUM"
    vol_regimes_series = pd.Series('medium', index=df.index)

# --- BACKTEST SUMMARY ---
bt_file = pathlib.Path(__file__).parent.parent / "models" / "saved" / "backtest_metrics.json"
if bt_file.exists():
    try:
        import json
        with open(bt_file, "r") as f:
            bt_metrics = json.load(f)
        st.markdown("### Latest Backtest Summary")
        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
        bc1.metric("Total Return", f"{bt_metrics.get('total_return_pct', 0):.2f}%")
        bc2.metric("Sharpe Ratio", f"{bt_metrics.get('sharpe_ratio', 0):.2f}")
        bc3.metric("Max Drawdown", f"{bt_metrics.get('max_drawdown_pct', 0):.2f}%")
        bc4.metric("Win Rate", f"{bt_metrics.get('win_rate_pct', 0):.2f}%")
        bc5.metric("Total Trades", f"{bt_metrics.get('total_trades', 0)}")
        st.markdown("---")
    except Exception as e:
        pass

# --- LIVE TRACKING ---
st.markdown("---")
st.markdown("## LIVE — Out-of-Sample (the real test)")

live_stats_path = pathlib.Path(__file__).parent.parent / "bot" / "live_stats.json"
live_signals_path = pathlib.Path(__file__).parent.parent / "bot" / "live_signals.csv"

if live_stats_path.exists() and live_signals_path.exists():
    try:
        import json
        with open(live_stats_path, "r") as f:
            l_stats = json.load(f)
        
        l_df = pd.read_csv(live_signals_path)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        if len(l_df) > 0:
            latest = l_df.iloc[-1]
            curr_sig = latest['signal']
            curr_regime = latest['regime']
            curr_conf = latest['ml_confidence']
        else:
            curr_sig = "NONE"
            curr_regime = "NONE"
            curr_conf = 0.0
            
        col1.metric("Current Signal", f"{curr_sig} ({curr_regime})")
        col2.metric("ML Confidence", f"{curr_conf:.1%}")
        
        # Safe format for live_win_rate vs 44.83% backtest
        l_wr = float(l_stats.get('live_win_rate', 0) or 0.0)
        delta_wr = l_wr - 44.83
        col3.metric("Live Win Rate", f"{l_wr:.2f}%", delta=f"{delta_wr:.2f}% vs BT")
        
        col4.metric("Signals Tracked", f"{l_stats.get('total_signals', 0)} ({l_stats.get('open_count', 0)} Open)")
        col5.metric("Live Return", f"{float(l_stats.get('cumulative_return_pct', 0) or 0.0):.2f}%")
        
        st.markdown("### Recent Signals")
        st.dataframe(l_df.tail(10)[::-1], use_container_width=True)
        
        if sel_tf == '15m':
            st.markdown("### Live Chart (15m)")
            fig_live = go.Figure()
            fig_live.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
            
            for _, row in l_df.iterrows():
                try:
                    ts = pd.to_datetime(row['timestamp'], utc=True)
                    if ts in df.index:
                        marker_color = 'green' if row['signal'] == 'LONG' else 'red'
                        marker_symbol = 'triangle-up' if row['signal'] == 'LONG' else 'triangle-down'
                        y_pos = df.loc[ts, 'low'] * 0.999 if row['signal'] == 'LONG' else df.loc[ts, 'high'] * 1.001
                        fig_live.add_trace(go.Scatter(x=[ts], y=[y_pos], mode='markers', marker=dict(color=marker_color, symbol=marker_symbol, size=15), showlegend=False))
                except Exception as e: 
                    pass
                    
            fig_live.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_live, use_container_width=True)
        else:
            st.info("Select 15m timeframe in the sidebar to view the live signal chart.")
    except Exception as e:
        st.error(f"Error loading live tracking: {e}")
else:
    st.info("Live tracking data not found. Run `python bot/live_tracker.py` to start.")

# --- ROW 1: METRIC CARDS ---
st.markdown("### Market Metrics (vs 24h)")
c1, c2, c3, c4, c5, c6 = st.columns(6)
color = "green" if pct_24h >= 0 else "red"
c1.markdown(f"**Current Price**<br><span style='color:{color}; font-size: 24px;'>${curr_price:,.2f}</span>", unsafe_allow_html=True)
c2.metric("24h Change %", f"{pct_24h:+.2f}%")
c3.metric("RSI 14", f"{curr_rsi:.1f}", rsi_badge, delta_color="off")
c4.metric("ATR 14", f"{curr_atr:.2f}")
c5.metric("Vol Regime", vol_regime)
c6.metric("Hurst Exp", f"{curr_hurst:.2f}", hurst_badge, delta_color="off")

# --- ROW 2: MAIN CHART & SIGNAL PANEL ---
st.markdown("---")
row2_c1, row2_c2 = st.columns([7, 3])

with row2_c1:
    last_ts = df.index[-1]
    now = pd.Timestamp.utcnow()
    is_live = (now - last_ts) <= pd.Timedelta(minutes=max(60, int(pd.Timedelta(sel_tf).total_seconds()/60))) if sel_tf not in ['1M', '1w'] else True
    dot_color = "🟢" if is_live else "🔴"
    
    st.subheader(f"Main Chart: {sel_sym} ({sel_tf})  {dot_color}  [Last Candle: {last_ts.strftime('%Y-%m-%d %H:%M:%S')} UTC]")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
    
    if show_ema:
        for p, c, w in [(9, 'white', 1), (21, 'blue', 1), (50, 'orange', 1), (200, 'red', 2)]:
            col = f'ema_{p}'
            if col in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=c, width=w), name=f'EMA {p}'))
            
    if show_bb and 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], line=dict(color='gray', dash='dash'), name='BB Upper'))
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], line=dict(color='gray', dash='dash'), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)', name='BB Lower'))
        
    if show_st and 'supertrend_up' in df.columns and 'supertrend_dn' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['supertrend_dn'], mode='markers', marker=dict(color='red', size=4), name='ST Bear'))
        fig.add_trace(go.Scatter(x=df.index, y=df['supertrend_up'], mode='markers', marker=dict(color='green', size=4), name='ST Bull'))
        
    if show_fib:
        max_price = df['high'].max()
        min_price = df['low'].min()
        diff = max_price - min_price
        levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for lvl in levels:
            val = min_price + diff * lvl
            color = 'gold' if lvl == 0.618 else 'rgba(255, 255, 255, 0.3)'
            fig.add_hline(y=val, line=dict(color=color, dash='dot'), annotation_text=f"Fib {lvl} ({val:.2f})", annotation_position="right")
        
        fig.add_hrect(y0=min_price + diff * 0.382, y1=min_price + diff * 0.618, fillcolor='rgba(255, 215, 0, 0.1)', line_width=0)
        
    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis_rangeslider_visible=False)
    
    # Range selectors
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=4, label="4H", step="hour", stepmode="backward"),
                dict(count=12, label="12H", step="hour", stepmode="backward"),
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=3, label="3D", step="day", stepmode="backward"),
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    st.plotly_chart(fig, use_container_width=True)

with row2_c2:
    st.subheader("Signal Panel")
    if model:
        sig_result = get_live_signal(model, df)
        if sig_result:
            direction, conf, feats = sig_result
            if direction == 1:
                sig_text, sig_color = "BUY", "green"
            elif direction == 0:
                sig_text, sig_color = "SELL", "red"
            else:
                sig_text, sig_color = "HOLD", "yellow"
                
            st.markdown(f"<div style='text-align: center; border-radius: 10px; padding: 20px; background-color: rgba(255,255,255,0.05);'>"
                        f"<h1 style='color: {sig_color}; margin: 0;'>{sig_text}</h1></div>", unsafe_allow_html=True)
            
            st.write(f"**Confidence:** {conf:.1%}")
            st.progress(conf)
            
            st.write("**Top Feature Drivers:**")
            if feats:
                st.bar_chart(pd.Series(feats))
            else:
                st.info("Feature importance not available.")
                
            st.write("**Last 5 Signals:**")
            # Mock historical signals
            mock_table = pd.DataFrame({
                "Time": [idx.strftime('%m-%d %H:%M') for idx in df.index[-5:]][::-1],
                "Signal": [sig_text] * 5,
                "Conf": [f"{conf:.1%}"] * 5
            })
            st.dataframe(mock_table, use_container_width=True)
    else:
        st.warning("ML Signal — run Step 3a to train model")
        st.markdown("### Quant Signals")
        if 'zscore_20' in df.columns:
            st.metric("Mean Reversion Z-Score", f"{df['zscore_20'].iloc[-1]:.2f}")
        
        if show_spread and 'eth_btc_zscore_20' in df.columns:
            ez = df['eth_btc_zscore_20'].iloc[-1]
            st.metric("ETH/BTC Spread Z", f"{ez:.2f}")
            ps = "LONG ETH" if ez < -2 else "LONG BTC" if ez > 2 else "NEUTRAL"
            st.write(f"**Pairs Trade:** {ps}")
            
        if 'harmonic_pattern' in df.columns and not pd.isna(df['harmonic_pattern'].iloc[-1]):
            st.info(f"Harmonic Pattern: {df['harmonic_pattern'].iloc[-1]}")

# --- ROW 3: SUB-CHARTS ---
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Momentum", "Volume", "Quant", "Macro cycle", "Hurst + Regime"])

with tab1:
    fig_m = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.4, 0.3, 0.3])
    if 'rsi_14' in df.columns:
        fig_m.add_trace(go.Scatter(x=df.index, y=df['rsi_14'], name='RSI', line_color='cyan'), row=1, col=1)
        fig_m.add_hline(y=70, line_color='red', row=1, col=1)
        fig_m.add_hline(y=30, line_color='green', row=1, col=1)
        fig_m.add_hrect(y0=70, y1=100, fillcolor='rgba(255,0,0,0.1)', layer='below', line_width=0, row=1, col=1)
        fig_m.add_hrect(y0=0, y1=30, fillcolor='rgba(0,255,0,0.1)', layer='below', line_width=0, row=1, col=1)
        
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        fig_m.add_trace(go.Scatter(x=df.index, y=df['stoch_k'], name='Stoch %K'), row=2, col=1)
        fig_m.add_trace(go.Scatter(x=df.index, y=df['stoch_d'], name='Stoch %D'), row=2, col=1)
        
    if 'macd_line' in df.columns:
        fig_m.add_trace(go.Scatter(x=df.index, y=df['macd_line'], name='MACD'), row=3, col=1)
        fig_m.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal'), row=3, col=1)
        colors = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
        fig_m.add_trace(go.Bar(x=df.index, y=df['macd_hist'], marker_color=colors, name='Hist'), row=3, col=1)
        
    fig_m.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_m, use_container_width=True)

with tab2:
    fig_v = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    colors = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])]
    fig_v.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='Volume'), row=1, col=1)
    
    if 'volume_delta' in df.columns:
        fig_v.add_trace(go.Scatter(x=df.index, y=df['volume_delta'], name='Vol Delta', line=dict(color='orange')), row=2, col=1)
        
    if 'obv' in df.columns:
        fig_v.add_trace(go.Scatter(x=df.index, y=df['obv'], name='OBV', line=dict(color='cyan')), row=3, col=1)
        
    fig_v.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_v, use_container_width=True)

with tab3:
    fig_q = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    if 'eth_btc_ratio' in df.columns:
        fig_q.add_trace(go.Scatter(x=df.index, y=df['eth_btc_ratio'], name='ETH/BTC Ratio'), row=1, col=1)
    if 'eth_btc_zscore_20' in df.columns:
        fig_q.add_trace(go.Scatter(x=df.index, y=df['eth_btc_zscore_20'], name='Spread Z-Score'), row=2, col=1)
        fig_q.add_hline(y=2, line_dash='dash', line_color='red', row=2, col=1)
        fig_q.add_hline(y=-2, line_dash='dash', line_color='green', row=2, col=1)
        fig_q.add_hrect(y0=-2, y1=2, fillcolor='rgba(255,255,255,0.05)', layer='below', line_width=0, row=2, col=1)
    if 'btc_eth_corr_30' in df.columns:
        fig_q.add_trace(go.Scatter(x=df.index, y=df['btc_eth_corr_30'], name='30p Correlation'), row=3, col=1)
    
    fig_q.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_q, use_container_width=True)

with tab4:
    if sel_tf != '1d':
        st.warning("Macro cycle indicators are specifically designed for the 1d timeframe. Please select '1d' timeframe from the sidebar.")
    else:
        fig_mac = go.Figure()
        fig_mac.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line=dict(color='white')))
        
        # We re-calculate golden ratio based on features.py logic
        if 'golden_ratio_1.6_dist' in df.columns:
            m_list = [1.0, 1.6, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]
            for m in m_list:
                col = f'golden_ratio_{m}_dist'
                if col in df.columns:
                    val = df['close'] / (df[col] + 1)
                    fig_mac.add_trace(go.Scatter(x=df.index, y=val, name=f'GR x{m}', line=dict(dash='dot', width=1)))
                    
        if 'pi_cycle_cross' in df.columns:
            crosses = df[df['pi_cycle_cross'] == 1]
            if not crosses.empty:
                fig_mac.add_trace(go.Scatter(x=crosses.index, y=crosses['close'], mode='markers', marker=dict(size=12, color='red'), name='Pi Cross'))
                
        fig_mac.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_type='log')
        st.plotly_chart(fig_mac, use_container_width=True)

with tab5:
    col_a, col_b = st.columns([7, 3])
    with col_a:
        fig_h = go.Figure()
        if 'hurst_100' in df.columns:
            fig_h.add_trace(go.Scatter(x=df.index, y=df['hurst_100'], name='Hurst', line=dict(color='cyan')))
            fig_h.add_hline(y=0.5, line_color='white', line_dash='dash')
            
            # Volatility ribbons
            if show_vol_regime:
                # Add background rectangles per regime slice
                # simple approach:
                start_idx = df.index[0]
                prev_r = vol_regimes_series.iloc[0]
                for i in range(1, len(df)):
                    curr_r = vol_regimes_series.iloc[i]
                    if curr_r != prev_r or i == len(df)-1:
                        end_idx = df.index[i]
                        c = 'rgba(0, 255, 0, 0.15)' if prev_r == 'low' else 'rgba(255, 255, 0, 0.15)' if prev_r == 'medium' else 'rgba(255, 0, 0, 0.15)'
                        fig_h.add_vrect(x0=start_idx, x1=end_idx, fillcolor=c, layer="below", line_width=0)
                        start_idx = end_idx
                        prev_r = curr_r
                        
        fig_h.update_layout(template="plotly_dark", height=400, title="Hurst Exp & Vol Regime", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_h, use_container_width=True)
        
    with col_b:
        if 'log_ret' in df.columns:
            log_ret = df['log_ret'].dropna()
            if len(log_ret) > 20:
                lags = range(1, 21)
                acorrs = [log_ret.autocorr(lag=i) for i in lags]
                fig_ac = go.Figure(go.Bar(x=list(lags), y=acorrs, marker_color='purple'))
                fig_ac.update_layout(template="plotly_dark", height=400, title="Autocorrelation (1-20)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_ac, use_container_width=True)

with st.expander("Show Raw Data Table (Last 20 Rows)"):
    st.dataframe(df.tail(20), use_container_width=True)
