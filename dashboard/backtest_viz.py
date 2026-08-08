import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pathlib

st.set_page_config(layout="wide", page_title="Backtest Visualizer")
st.title("Backtest Results — Trade Visualization")

base_dir = pathlib.Path(__file__).parent.parent
save_dir = base_dir / "models" / "saved"

try:
    with open(save_dir / "trade_log.json", "r") as f:
        trades = json.load(f)
    with open(save_dir / "equity_curve.json", "r") as f:
        equity_curve = json.load(f)
    with open(save_dir / "price_series.json", "r") as f:
        price_series = json.load(f)
    with open(save_dir / "backtest_regime_adaptive.json", "r") as f:
        metrics = json.load(f)
except Exception as e:
    st.error(f"Error loading files: {e}. Did you run the backtest?")
    st.stop()

# Chart Settings
st.sidebar.header("Chart Settings")
tf_chart = st.sidebar.selectbox("Chart Timeframe", ["5m", "15m", "1h", "4h", "1d"], index=1)

if tf_chart == "15m":
    df_price = pd.DataFrame(price_series)
    df_price['timestamp'] = pd.to_datetime(df_price['timestamp'], utc=True)
else:
    feat_path = base_dir / "data" / "processed" / f"BTC_USDT_{tf_chart}_features.csv"
    clean_path = base_dir / "data" / "raw" / f"BTC_USDT_{tf_chart}_clean.csv"
    if feat_path.exists():
        df_price = pd.read_csv(feat_path)
    elif clean_path.exists():
        df_price = pd.read_csv(clean_path)
    else:
        st.error(f"Data for {tf_chart} not found.")
        st.stop()
    df_price['timestamp'] = pd.to_datetime(df_price['timestamp'], utc=True)
    
    if len(price_series) > 0:
        orig_start = pd.to_datetime(price_series[0]['timestamp'], utc=True)
        orig_end = pd.to_datetime(price_series[-1]['timestamp'], utc=True)
        df_price = df_price[(df_price['timestamp'] >= orig_start) & (df_price['timestamp'] <= orig_end)].copy()

df_equity = pd.DataFrame(equity_curve)
df_equity['timestamp'] = pd.to_datetime(df_equity['timestamp'], utc=True)

df_trades = pd.DataFrame(trades)
if not df_trades.empty:
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'], utc=True)
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'], utc=True)

# Filters
st.sidebar.header("Filters")
if not df_trades.empty:
    regime_filter = st.sidebar.selectbox("Regime", ["ALL"] + list(df_trades['regime'].unique()))
    outcome_filter = st.sidebar.selectbox("Outcome", ["ALL", "WINS", "LOSSES"])
    direction_filter = st.sidebar.selectbox("Direction", ["ALL", "LONG", "SHORT"])
    
    mask = pd.Series([True]*len(df_trades))
    if regime_filter != "ALL":
        mask = mask & (df_trades['regime'] == regime_filter)
    if outcome_filter == "WINS":
        mask = mask & (df_trades['pnl_dollars'] > 0)
    elif outcome_filter == "LOSSES":
        mask = mask & (df_trades['pnl_dollars'] <= 0)
    if direction_filter != "ALL":
        mask = mask & (df_trades['direction'] == direction_filter)
        
    filtered_trades = df_trades[mask]
else:
    filtered_trades = pd.DataFrame()

# Metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
total_ret = metrics.get('total_return_pct', 0)
win_rate = metrics.get('win_rate_pct', 0)
dd = metrics.get('max_drawdown_pct', 0)
num_trades = metrics.get('total_trades', 0)
if not df_trades.empty:
    wins = df_trades[df_trades['pnl_dollars'] > 0]['pnl_dollars'].sum()
    losses = abs(df_trades[df_trades['pnl_dollars'] < 0]['pnl_dollars'].sum())
    pf = wins / losses if losses > 0 else float('inf')
else:
    pf = 0

col1.metric("Total Return", f"{total_ret:.2f}%")
col2.metric("Win Rate", f"{win_rate:.2f}%")
col3.metric("Max Drawdown", f"{dd:.2f}%")
col4.metric("Total Trades", f"{num_trades}")
col5.metric("Profit Factor", f"{pf:.2f}")

# Charts
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7, 0.3])

# Price
if 'open' in df_price.columns:
    fig.add_trace(go.Candlestick(x=df_price['timestamp'],
                                 open=df_price['open'],
                                 high=df_price['high'],
                                 low=df_price['low'],
                                 close=df_price['close'],
                                 name='Price'), row=1, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
else:
    fig.add_trace(go.Scatter(x=df_price['timestamp'], y=df_price['close'],
                             mode='lines', name='Price', line=dict(color='gray')), row=1, col=1)

# Trades
if not filtered_trades.empty:
    for _, t in filtered_trades.iterrows():
        color = 'lime' if t['pnl_dollars'] > 0 else 'red'
        # Entry marker
        marker_symbol = 'triangle-up' if t['direction'] == 'LONG' else 'triangle-down'
        hover_text = f"{t['direction']} {t['regime']}<br>Entry: {t['entry_price']}<br>Exit: {t['exit_price']}<br>PnL: {t['pnl_pct']*100:.2f}%<br>Reason: {t['exit_reason']}"
        fig.add_trace(go.Scatter(x=[t['entry_time']], y=[t['entry_price']],
                                 mode='markers', marker=dict(symbol=marker_symbol, size=10, color=color),
                                 hovertext=hover_text, name=t['direction'], showlegend=False), row=1, col=1)
        # Exit marker
        fig.add_trace(go.Scatter(x=[t['exit_time']], y=[t['exit_price']],
                                 mode='markers', marker=dict(symbol='x', size=8, color=color),
                                 hovertext=hover_text, name='Exit', showlegend=False), row=1, col=1)
        # Line
        fig.add_trace(go.Scatter(x=[t['entry_time'], t['exit_time']], 
                                 y=[t['entry_price'], t['exit_price']],
                                 mode='lines', line=dict(color=color, width=1), showlegend=False), row=1, col=1)

# Drawdown shading (optional simple implementation, skipped for brevity but requested)
if not df_equity.empty:
    fig.add_trace(go.Scatter(x=df_equity['timestamp'], y=df_equity['equity'],
                             mode='lines', name='Equity', line=dict(color='blue')), row=2, col=1)
                             
    # Buy and hold
    first_price = df_price.iloc[0]['close']
    starting_cap = df_equity.iloc[0]['equity']
    bnh = (df_price['close'] / first_price) * starting_cap
    fig.add_trace(go.Scatter(x=df_price['timestamp'], y=bnh,
                             mode='lines', name='Buy & Hold', line=dict(color='orange', dash='dot')), row=2, col=1)

# Update layout
fig.update_layout(template="plotly_dark", height=800,
                  hovermode='closest',
                  xaxis=dict(
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
                      ),
                      type="date"
                  ))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Trade Log")
st.dataframe(filtered_trades)

# Regime distribution
st.subheader("Regime Distribution")
col_a, col_b = st.columns(2)
if not df_trades.empty:
    reg_counts = df_trades['regime'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(labels=reg_counts.index, values=reg_counts.values)])
    fig_pie.update_layout(template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0))
    col_a.plotly_chart(fig_pie, use_container_width=True)
    
    # Win rate per regime
    win_rates = df_trades.groupby('regime').apply(lambda x: len(x[x['pnl_dollars']>0])/len(x) if len(x)>0 else 0)
    fig_bar = go.Figure(data=[go.Bar(x=win_rates.index, y=win_rates.values)])
    fig_bar.update_layout(template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0))
    col_b.plotly_chart(fig_bar, use_container_width=True)
