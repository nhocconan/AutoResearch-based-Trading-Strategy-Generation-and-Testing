#!/usr/bin/env python3
"""
6h Donchian(20) Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: Price breaking above/below 6h Donchian Channel (20-period high/low) in the direction 
of weekly pivot (above/below weekly pivot point) with volume confirmation (volume > 1.5x average) 
indicates strong momentum. Weekly pivot provides higher timeframe bias, reducing counter-trend trades.
Designed for both bull and bear markets by using pivot direction as filter.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot point (using 1w data as HTF reference)
    df_1w = get_htf_data(prices, '1w')
    # Typical price = (H + L + C) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Weekly pivot = (Prior week H + L + C) / 3
    weekly_pivot = typical_price.shift(1).values  # Shift to avoid look-ahead
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators (max of 20,20,20)
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_conf = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly pivot with volume
            if price > upper and price > weekly_pivot_val and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below weekly pivot with volume
            elif price < lower and price < weekly_pivot_val and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to weekly pivot or Donchian low
            if price < weekly_pivot_val or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to weekly pivot or Donchian high
            if price > weekly_pivot_val or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0