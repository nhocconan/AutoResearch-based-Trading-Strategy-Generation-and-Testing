#!/usr/bin/env python3
"""
12h_Weekly_Donchian_Breakout_Volume_Filter
Long when price breaks above weekly Donchian high (20) + volume surge + price above daily EMA50.
Short when price breaks below weekly Donchian low (20) + volume surge + price below daily EMA50.
Exit when price returns to weekly Donchian mid-point.
Designed to capture strong weekly trends with volume confirmation, avoiding chop.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Donchian Channel (20) ===
    df_1w = get_htf_data(prices, '1w')
    donch_high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high_20 + donch_low_20) / 2.0
    
    # Align weekly Donchian levels to 12h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1w, donch_mid)
    
    # === Daily EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume surge filter (2x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly Donchian high, volume surge, price above daily EMA50
            if (close[i] > donch_high_20_aligned[i] and 
                volume_surge[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly Donchian low, volume surge, price below daily EMA50
            elif (close[i] < donch_low_20_aligned[i] and 
                  volume_surge[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to weekly Donchian mid-point
        elif position == 1:
            # Exit long: price crosses below weekly Donchian mid-point
            if close[i] < donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly Donchian mid-point
            if close[i] > donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Weekly_Donchian_Breakout_Volume_Filter"
timeframe = "12h"
leverage = 1.0