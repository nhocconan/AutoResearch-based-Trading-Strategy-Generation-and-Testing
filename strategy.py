#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Williams %R breakout + volume confirmation + 1w EMA50 filter.
Long when price breaks above Williams %R oversold (-80) with volume > 1.3x 20-period average and close > 1w EMA50.
Short when price breaks below Williams %R overbought (-20) with volume > 1.3x 20-period average and close < 1w EMA50.
Williams %R captures mean reversion extremes that work in both trending and ranging markets.
Volume confirmation reduces false signals. 1w EMA50 ensures we trade with the higher timeframe trend.
Target: 50-150 total trades over 4 years (12-37/year) to avoid overtrading.
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
    
    # Get 1d data for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Williams %R (14)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1d volume 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA50
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(volume_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: Williams %R breaks above oversold (-80) with volume and uptrend filter
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                volume_confirmed and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R breaks below overbought (-20) with volume and downtrend filter
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  volume_confirmed and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R reaches overbought (-20) or trend filter fails
            if (williams_r_aligned[i] >= -20 or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R reaches oversold (-80) or trend filter fails
            if (williams_r_aligned[i] <= -80 or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dWilliamsR_Volume_1wEMA50"
timeframe = "6h"
leverage = 1.0