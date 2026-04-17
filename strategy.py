#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1w Williams %R extreme + 1d EMA trend filter + volume spike.
Long when 1w Williams %R < -80 (oversold) AND price > 1d EMA50 AND 1d volume > 2.0x 20-period average.
Short when 1w Williams %R > -20 (overbought) AND price < 1d EMA50 AND 1d volume > 2.0x 20-period average.
Exit when Williams %R returns to -50 level or volume drops below average.
Williams %R captures exhaustion points that work in both bull and bear markets.
EMA50 filter ensures we trade with the intermediate-term trend.
Volume spike confirms conviction behind the move.
Discrete position sizing of 0.25 to limit fee drag and manage drawdown.
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
    
    # Get 1w data for Williams %R
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R (14)
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1w) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Get 1d data for EMA and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price above EMA50 + volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + price below EMA50 + volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 or volume drops below average
            if (williams_r_aligned[i] > -50 or 
                volume_1d_aligned[i] < vol_ma_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 or volume drops below average
            if (williams_r_aligned[i] < -50 or 
                volume_1d_aligned[i] < vol_ma_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1wWilliamsR_EMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0