#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 1w EMA50 trend filter and volume confirmation
# Long when 1d Williams %R < -80 (oversold) AND price > 1w EMA50 (bullish trend) AND volume > 1.5 * avg_volume(20) on 6h
# Short when 1d Williams %R > -20 (overbought) AND price < 1w EMA50 (bearish trend) AND volume > 1.5 * avg_volume(20) on 6h
# Exit when Williams %R crosses back through -50 (mean reversion)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe
# Williams %R identifies exhaustion points in both bull and bear markets
# 1w EMA50 filter ensures we trade with the higher timeframe trend, reducing counter-trend whipsaw
# Volume confirmation validates the exhaustion move with participation

name = "6h_1dWilliamsR_Extreme_1wEMA50_VolumeConfirm"
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
    if len(df_1d) < 14:  # Need at least 14 completed daily bars for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)
    # Handle division by zero (when high == low)
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need at least 50 completed weekly bars for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA (50-period)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80), price above 1w EMA50 (bullish trend), volume confirmation, in session
            if (williams_r_1d_aligned[i] < -80.0 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), price below 1w EMA50 (bearish trend), volume confirmation, in session
            elif (williams_r_1d_aligned[i] > -20.0 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (mean reversion)
            if williams_r_1d_aligned[i] > -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (mean reversion)
            if williams_r_1d_aligned[i] < -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals