#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout_Volume_Trend_v1
Hypothesis: 1d timeframe with 1w Camarilla pivot levels, volume confirmation, and 1w EMA trend filter.
Designed for 10-30 trades/year by requiring breakouts of H3/L3 levels with volume > 1.5x average
and price aligned with 1w EMA trend. Works in bull/bear markets by only taking trend-aligned breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Pivot_Breakout_Volume_Trend_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for Camarilla pivots and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + range_hl * 1.1 / 4
    L3 = pivot - range_hl * 1.1 / 4
    H4 = pivot + range_hl * 1.1 / 2
    L4 = pivot - range_hl * 1.1 / 2
    
    # Align to 1d timeframe
    H3_1d = align_htf_to_ltf(prices, df_1w, H3)
    L3_1d = align_htf_to_ltf(prices, df_1w, L3)
    H4_1d = align_htf_to_ltf(prices, df_1w, H4)
    L4_1d = align_htf_to_ltf(prices, df_1w, L4)
    
    # Calculate 1w EMA (21 period) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume average (20 period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_1d[i]) or np.isnan(L3_1d[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter: price above/below 1w EMA
        above_ema = close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1w_aligned[i]
        
        # Entry conditions: breakout of H3/L3 with volume and trend
        long_entry = (close[i] > H3_1d[i]) and volume_spike and above_ema
        short_entry = (close[i] < L3_1d[i]) and volume_spike and below_ema
        
        # Exit conditions: return to H4/L4 levels or trend reversal
        long_exit = (close[i] < H4_1d[i]) or (close[i] < ema_1w_aligned[i])
        short_exit = (close[i] > L4_1d[i]) or (close[i] > ema_1w_aligned[i])
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals