#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Breakout_Volume_Trend_v1
Hypothesis: 4h timeframe with 12h Camarilla pivot levels, volume confirmation, and 12h EMA trend filter.
Designed for fewer trades (target 20-50/year) by requiring breakouts of H3/L3 levels with volume > 1.5x average
and price aligned with 12h EMA trend. Works in bull/bear markets by only taking trend-aligned breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Pivot_Breakout_Volume_Trend_v1"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop for Camarilla pivots and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h period
    prev_close = df_12h['close'].shift(1).values
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + range_hl * 1.1 / 4
    L3 = pivot - range_hl * 1.1 / 4
    H4 = pivot + range_hl * 1.1 / 2
    L4 = pivot - range_hl * 1.1 / 2
    
    # Align to 4h timeframe
    H3_4h = align_htf_to_ltf(prices, df_12h, H3)
    L3_4h = align_htf_to_ltf(prices, df_12h, L3)
    H4_4h = align_htf_to_ltf(prices, df_12h, H4)
    L4_4h = align_htf_to_ltf(prices, df_12h, L4)
    
    # Calculate 12h EMA (21 period) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume average (20 period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter: price above/below 12h EMA
        above_ema = close[i] > ema_12h_aligned[i]
        below_ema = close[i] < ema_12h_aligned[i]
        
        # Entry conditions: breakout of H3/L3 with volume and trend
        long_entry = (close[i] > H3_4h[i]) and volume_spike and above_ema
        short_entry = (close[i] < L3_4h[i]) and volume_spike and below_ema
        
        # Exit conditions: return to H4/L4 levels or trend reversal
        long_exit = (close[i] < H4_4h[i]) or (close[i] < ema_12h_aligned[i])
        short_exit = (close[i] > L4_4h[i]) or (close[i] > ema_12h_aligned[i])
        
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