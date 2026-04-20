#!/usr/bin/env python3
# 1d_1w_Donchian20_Breakout_Volume_Confirmation
# Hypothesis: Buy when price breaks above 1d Donchian(20) high with volume confirmation and 1w uptrend (price > 1w EMA200). 
# Sell/short when price breaks below 1d Donchian(20) low with volume confirmation and 1w downtrend (price < 1w EMA200).
# Uses 1w trend filter to avoid counter-trend trades. Target: 30-100 trades over 4 years (7-25/year).
# Works in bull/bear via 1w trend filter - only trade with the 1w trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Donchian20_Breakout_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # === 1w: EMA200 for trend filter ===
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # === 1d: Donchian(20) channels ===
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Calculate rolling max/min for Donchian channels
    high_max20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # === 1d: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Get values
        close_val = close_1d[i]
        high_val = high_1d[i]
        low_val = low_1d[i]
        ema200_1w_val = ema200_1w_aligned[i]
        high_max20_val = high_max20[i]
        low_min20_val = low_min20[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema200_1w_val) or np.isnan(high_max20_val) or np.isnan(low_min20_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian(20) high with volume confirmation and 1w uptrend
            if (high_val > high_max20_val and  # Price broke above Donchian high
                vol_ratio_val > 2.0 and        # Volume confirmation
                close_val > ema200_1w_val):    # 1w uptrend filter
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below Donchian(20) low with volume confirmation and 1w downtrend
            elif (low_val < low_min20_val and   # Price broke below Donchian low
                  vol_ratio_val > 2.0 and       # Volume confirmation
                  close_val < ema200_1w_val):   # 1w downtrend filter
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian(20) low or trend reversal
            if low_val < low_min20_val or close_val < ema200_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: Price breaks above Donchian(20) high or trend reversal
            if high_val > high_max20_val or close_val > ema200_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals