#!/usr/bin/env python3
"""
1d_Weekly_Range_Breakout_v1
Hypothesis: Buy near weekly lows during accumulation, sell near weekly highs during distribution.
Uses weekly range to identify accumulation/distribution phases. In both bull and bear markets,
price tends to respect weekly support/resistance levels. Adds volume confirmation to avoid
false breakouts. Target: 20-50 trades over 4 years (5-12/year) on 1d timeframe.
"""

name = "1d_Weekly_Range_Breakout_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Data for Range Calculation ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    # Previous week's range (avoid look-ahead)
    prev_high_1w = high_1w  # This is the prior week's high
    prev_low_1w = low_1w    # This is the prior week's low
    
    # Align weekly range to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, prev_high_1w)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, prev_low_1w)
    
    # === Volume Confirmation (Daily) ===
    # Volume ratio: current volume vs 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price near weekly low with volume confirmation (accumulation)
            # Near weekly low: within 1.5% of weekly low
            near_weekly_low = (low[i] <= weekly_low_aligned[i] * 1.015)
            # Volume confirmation: above average volume
            vol_confirm = vol_ratio[i] > 1.2
            
            if near_weekly_low and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price near weekly high with volume confirmation (distribution)
            # Near weekly high: within 1.5% of weekly high
            near_weekly_high = (high[i] >= weekly_high_aligned[i] * 0.985)
            if near_weekly_high and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches weekly high or loses momentum
            # Exit when price reaches weekly high or volume dries up
            if (high[i] >= weekly_high_aligned[i] * 0.995) or (vol_ratio[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price reaches weekly low or loses momentum
            # Exit when price reaches weekly low or volume dries up
            if (low[i] <= weekly_low_aligned[i] * 1.005) or (vol_ratio[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals