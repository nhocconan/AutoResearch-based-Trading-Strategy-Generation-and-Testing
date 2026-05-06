#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extreme readings combined with 1w EMA34 trend filter and volume spike confirmation
# Long when 1d Williams %R < -80 (oversold) AND 1w EMA34 is rising AND 6h volume > 2.0 * avg_volume(20)
# Short when 1d Williams %R > -20 (overbought) AND 1w EMA34 is falling AND 6h volume > 2.0 * avg_volume(20)
# Exit when Williams %R returns to -50 midpoint or opposite extreme
# Williams %R identifies exhaustion points in both bull and bear markets
# 1w EMA34 ensures we trade with the weekly trend while reducing whipsaws
# Volume spike confirms conviction at exhaustion points
# Target: 75-150 total trades over 4 years (19-38/year) for 6h timeframe

name = "6h_1dWilliamsR_Extreme_1wEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 completed daily bars for Williams %R(14)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need at least 34 completed weekly bars for EMA34
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80), EMA34 rising, volume spike
            if (williams_r_1d_aligned[i] < -80.0 and 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), EMA34 falling, volume spike
            elif (williams_r_1d_aligned[i] > -20.0 and 
                  ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 midpoint or above -20 (overbought)
            if williams_r_1d_aligned[i] >= -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 midpoint or below -80 (oversold)
            if williams_r_1d_aligned[i] <= -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals