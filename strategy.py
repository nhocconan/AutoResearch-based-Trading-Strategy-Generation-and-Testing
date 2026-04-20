#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1-day ATR-based volatility filter
# Williams %R identifies overbought/oversold conditions; ATR filter avoids trading during low volatility
# Long when Williams %R < -80 and ATR ratio > 1.2; Short when Williams %R > -20 and ATR ratio > 1.2
# Uses 14-period Williams %R on 12h and 14-period ATR ratio (current/20-period average) on 1d
# Designed to work in both bull and bear markets by fading extremes only when volatility is sufficient
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for Williams %R
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 14-period Williams %R on 12h timeframe
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close_12h) / (highest_high - lowest_low + 1e-10)
    willr_12h_aligned = align_htf_to_ltf(prices, df_12h, willr)
    
    # Load daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on daily timeframe
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average ATR for volatility comparison
    atr_20_avg = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR ratio (current ATR / 20-period average ATR)
    atr_ratio = atr_14 / (atr_20_avg + 1e-10)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(willr_12h_aligned[i]) or np.isnan(atr_ratio_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        willr_val = willr_12h_aligned[i]
        vol_filter = atr_ratio_aligned[i] > 1.2  # Trade only when volatility is above average
        
        if position == 0:
            # Enter long conditions: Williams %R oversold + volatility filter
            if willr_val < -80 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short conditions: Williams %R overbought + volatility filter
            elif willr_val > -20 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral or volatility drops
            if willr_val > -50 or atr_ratio_aligned[i] <= 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral or volatility drops
            if willr_val < -50 or atr_ratio_aligned[i] <= 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_ATRVolFilter"
timeframe = "12h"
leverage = 1.0