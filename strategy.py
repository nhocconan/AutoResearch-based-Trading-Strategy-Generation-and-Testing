#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
    # Long: price breaks above H3 (resistance) AND price > 4h EMA50 AND 08-20 UTC session
    # Short: price breaks below L3 (support) AND price < 4h EMA50 AND 08-20 UTC session
    # Exit: price returns to pivot point (mean reversion)
    # Using 4h for trend (EMA50) and 1d for Camarilla pivots (structure), 1h only for entry timing
    # Discrete position sizing (0.20) to minimize fee drag
    # Target: 15-37 trades/year (~60-150 over 4 years) to avoid overtrading
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute once)
    hours = open_time.dt.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivots (based on previous 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # PIVOT = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # RANGE = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels:
    # H3 = C + RANGE * 1.1/4
    # L3 = C - RANGE * 1.1/4
    h3_1d = close_1d + range_1d * 1.1 / 4
    l3_1d = close_1d - range_1d * 1.1 / 4
    
    # Align 1d Camarilla levels to 1h (wait for completed 1d bar)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Get 4h data for EMA50 trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema_4h_aligned[i]) or
            not in_session.iloc[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: only long if price > 4h EMA50, only short if price < 4h EMA50
        long_trend_ok = close[i] > ema_4h_aligned[i]
        short_trend_ok = close[i] < ema_4h_aligned[i]
        
        # Entry logic: Camarilla breakout + trend + session
        long_entry = (close[i] > h3_1d_aligned[i]) and long_trend_ok
        short_entry = (close[i] < l3_1d_aligned[i]) and short_trend_ok
        
        # Exit logic: return to pivot (mean reversion)
        long_exit = close[i] < pivot_1d_aligned[i]
        short_exit = close[i] > pivot_1d_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_1d_4h_camarilla_breakout_trend_session_v1"
timeframe = "1h"
leverage = 1.0