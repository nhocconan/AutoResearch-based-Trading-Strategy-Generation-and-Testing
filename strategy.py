#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Pivot_Breakout_Volume_Filter_v1
Hypothesis: Use 4h and 1d timeframes for directional bias and structure, with 1h for entry timing.
Buy when price breaks above 1d H3 with volume spike and 4h trend alignment; sell when breaks below 1d L3 with volume spike and 4h trend alignment.
Only trade during active session (08-20 UTC) to avoid low-liquidity hours.
Target: 15-30 trades/year by requiring volume > 2x average and strict trend filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Pivot_Breakout_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + range_hl * 1.1 / 4
    L3 = pivot - range_hl * 1.1 / 4
    H4 = pivot + range_hl * 1.1 / 2
    L4 = pivot - range_hl * 1.1 / 2
    
    # Align to 1h timeframe
    H3_1h = align_htf_to_ltf(prices, df_1d, H3)
    L3_1h = align_htf_to_ltf(prices, df_1d, L3)
    H4_1h = align_htf_to_ltf(prices, df_1d, H4)
    L4_1h = align_htf_to_ltf(prices, df_1d, L4)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA (34 period) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_1h = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume average (20 period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):
        # Skip if outside trading session or missing data
        if not in_session[i] or \
           np.isnan(H3_1h[i]) or np.isnan(L3_1h[i]) or \
           np.isnan(ema_4h_1h[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume spike: current volume > 2x average (strict filter)
        volume_spike = volume[i] > vol_ma[i] * 2.0
        
        # Trend filter: price above/below 4h EMA
        above_ema = close[i] > ema_4h_1h[i]
        below_ema = close[i] < ema_4h_1h[i]
        
        # Entry conditions: breakout of H3/L3 with volume and trend
        long_entry = (close[i] > H3_1h[i]) and volume_spike and above_ema
        short_entry = (close[i] < L3_1h[i]) and volume_spike and below_ema
        
        # Exit conditions: return to H4/L4 levels
        long_exit = close[i] < H4_1h[i]
        short_exit = close[i] > L4_1h[i]
        
        # Priority: entry > exit > hold
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
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals