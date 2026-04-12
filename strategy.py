#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Breakout_v1
Hypothesis: Use 4h Camarilla H3/L3 for breakout direction with 1d trend filter, 
and 1h for precise entry timing with volume confirmation. 
Targets 15-37 trades/year by requiring multiple confluence factors.
Works in bull/bear via 1d trend filter and mean-reversion exit.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H CAMARILLA LEVELS ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels using previous day's close
    close_prev = np.roll(close_4h, 1)
    close_prev[0] = close_4h[0]  # first value
    range_4h = high_4h - low_4h
    
    h3 = close_prev + (range_4h * 1.1 / 4)
    l3 = close_prev - (range_4h * 1.1 / 4)
    h4 = close_prev + (range_4h * 1.1)
    l4 = close_prev - (range_4h * 1.1)
    
    # Align 4h levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4)
    
    # === 1D TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA50 for trend direction
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_1d > ema50
    trend_down = close_1d < ema50
    
    # Align trend to 1h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down.astype(float))
    
    # === 1H VOLUME CONFIRMATION ===
    # Volume average (20-period)
    vol_sum = 0.0
    vol_count = 0
    vol_avg = np.zeros(n)
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / max(vol_count, 1)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not in session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if indicators not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Entry conditions
        long_setup = (close[i] > h3_aligned[i]) and vol_confirm and trend_up_aligned[i] > 0.5
        short_setup = (close[i] < l3_aligned[i]) and vol_confirm and trend_down_aligned[i] > 0.5
        
        # Exit conditions: mean reversion to opposite H4/L4 levels
        exit_long = close[i] < l4_aligned[i]
        exit_short = close[i] > h4_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals