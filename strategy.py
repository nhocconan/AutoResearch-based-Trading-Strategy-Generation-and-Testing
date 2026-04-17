#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Williams %R extremes and volume confirmation.
Trade reversals from oversold/overbought conditions using 12h Williams %R(14).
Enter long when Williams %R < -80 (oversold) with volume spike (>2.0x 20-period average).
Enter short when Williams %R > -20 (overbought) with volume spike.
Use 12h ADX > 20 to avoid ranging markets and reduce whipsaws.
Position sizing: 0.25 for entries, 0 for exits.
Target: 75-200 total trades over 4 years (19-50/year).
Williams %R captures momentum exhaustion; volume confirms institutional interest.
Works in both bull (buy dips) and bear (sell rallies) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R and ADX
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Williams %R (14)
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 12h ADX (14)
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 2.0x 20-period average
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80), volume spike, avoid strong downtrend
            if (williams_r_aligned[i] < -80 and 
                volume[i] > vol_ma_20_aligned[i] * 2.0 and 
                adx_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), volume spike, avoid strong uptrend
            elif (williams_r_aligned[i] > -20 and 
                  volume[i] > vol_ma_20_aligned[i] * 2.0 and 
                  adx_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns above -50 or volume drops
            if williams_r_aligned[i] > -50 or volume[i] < vol_ma_20_aligned[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns below -50 or volume drops
            if williams_r_aligned[i] < -50 or volume[i] < vol_ma_20_aligned[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hWilliamsR_Volume_ADX"
timeframe = "4h"
leverage = 1.0