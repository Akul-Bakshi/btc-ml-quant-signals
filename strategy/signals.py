import pandas as pd

def detect_regime(df_4h):
    """Classifies market as TREND, RANGE, or CHOP."""
    trending_score = 0
    ranging_score = 0
    
    adx = df_4h['adx_14'].iloc[-1]
    if adx > 25: trending_score += 1
    elif adx < 20: ranging_score += 1
    
    ema50 = df_4h['ema_50'].iloc[-1]
    ema200 = df_4h['ema_200'].iloc[-1]
    sep = abs(ema50 - ema200) / ema200
    if sep > 0.03: trending_score += 1
    elif sep < 0.01: ranging_score += 1
    
    bw = df_4h['bb_bandwidth'].iloc[-1]
    bw_ma = df_4h['bb_bandwidth'].rolling(20).mean().iloc[-1]
    if bw > bw_ma * 1.2: trending_score += 1
    elif bw < bw_ma * 0.8: ranging_score += 1
    
    roc = df_4h['roc_10'].iloc[-1]
    if abs(roc) > 3.0: trending_score += 1
    elif abs(roc) < 1.0: ranging_score += 1
    
    if trending_score >= 2 and trending_score > ranging_score:
        regime = 'TREND'
        direction = 'LONG' if ema50 > ema200 else 'SHORT'
    elif ranging_score >= 2 and ranging_score > trending_score:
        regime = 'RANGE'
        direction = None  # decided at entry
    else:
        regime = 'CHOP'
        direction = None
    
    return {'regime': regime, 'direction': direction,
            'trending_score': trending_score,
            'ranging_score': ranging_score, 'adx': adx}

def get_trend_entry(df_15m, df_1h, trend_direction):
    """Entry logic for TREND regime. Returns dict or None."""
    c15 = df_15m.iloc[-1]
    c1h = df_1h.iloc[-1]
    
    if trend_direction == 'LONG':
        if not (c1h['ema_21'] > c1h['ema_50']): return None
        if not (c1h['macd_hist'] > 0): return None
        if not (c15['close'] > c15['ema_21']): return None
        if not (c15['volume_zscore_20'] > 0.2): return None
        if not (45 <= c15['rsi_14'] <= 65): return None
    else:  # SHORT
        if not (c1h['ema_21'] < c1h['ema_50']): return None
        if not (c1h['macd_hist'] < 0): return None
        if not (c15['close'] < c15['ema_21']): return None
        if not (c15['volume_zscore_20'] > 0.2): return None
        if not (35 <= c15['rsi_14'] <= 55): return None
    
    return {'direction': trend_direction, 'entry_type': 'TREND'}

def get_range_entry(df_15m, df_4h):
    """Entry logic for RANGE regime. Returns dict or None."""
    c15 = df_15m.iloc[-1]
    c4h = df_4h.iloc[-1]
    price = c15['close']
    
    # Check fibonacci touch within 0.8%
    fib_levels = {
        '0.382': c4h['fib_0.382'],
        '0.5': c4h['fib_0.5'],
        '0.618': c4h['fib_0.618']
    }
    touched = None
    for name, level in fib_levels.items():
        if abs(price - level) / price < 0.012:
            touched = (name, level)
            break
    if touched is None: return None
    
    cond_rsi_long = c15['rsi_7'] < 35
    cond_rsi_short = c15['rsi_7'] > 65
    
    cond_zscore_long = c15['zscore_20'] < -1.2
    cond_zscore_short = c15['zscore_20'] > 1.2
    
    cond_volume = c15['volume_zscore_20'] > 0.15
    
    long_count = int(cond_rsi_long) + int(cond_zscore_long) + int(cond_volume)
    short_count = int(cond_rsi_short) + int(cond_zscore_short) + int(cond_volume)
    
    if long_count >= 1 and long_count > short_count:
        direction = 'LONG'
    elif short_count >= 1 and short_count > long_count:
        direction = 'SHORT'
    else:
        return None
    
    return {'direction': direction, 'entry_type': 'RANGE',
            'fib_level': touched[0], 'fib_price': touched[1]}

def get_combined_signal(df_15m, df_1h, df_4h, model, features):
    """Master signal function. Returns signal dict."""
    regime_data = detect_regime(df_4h)
    regime = regime_data['regime']
    
    if regime == 'CHOP':
        return {'signal': 'NO_TRADE', 'reason': 'CHOP',
                'regime': 'CHOP'}
    
    if regime == 'TREND':
        entry = get_trend_entry(df_15m, df_1h,
                                regime_data['direction'])
        stop_pct, target_pct = 0.008, 0.016
    else:  # RANGE
        entry = get_range_entry(df_15m, df_4h)
        stop_pct, target_pct = 0.004, 0.010
    
    if entry is None:
        return {'signal': 'NO_TRADE', 'reason': 'no entry',
                'regime': regime}
    
    direction = entry['direction']
    
    # ML confirmation
    try:
        row = df_15m.iloc[-1][features].values.reshape(1, -1)
        proba = model.predict_proba(row)[0]
        confidence = proba[1] if direction == 'LONG' else proba[0]
    except Exception as e:
        confidence = 0.0
    
    import yaml
    with open("config.yaml", "r") as f:
        conf = yaml.safe_load(f)
    thresh = conf.get('trading', {}).get('ml_confidence_threshold', 0.55)
    
    if confidence < thresh:
        return {'signal': 'NO_TRADE',
                'reason': f'ML conf {confidence:.2f}',
                'regime': regime}
    
    return {'signal': direction, 'regime': regime,
            'confidence': confidence, 'stop_pct': stop_pct,
            'target_pct': target_pct, 'entry_type': entry['entry_type']}
