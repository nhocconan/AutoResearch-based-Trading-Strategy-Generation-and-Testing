#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h/1d confluence with Williams %R and volume confirmation.
# Williams %R identifies overbought/oversold conditions on 12h timeframe.
# Long when 12h Williams %R < -80 (oversold) + price above 1d EMA50 + volume spike.
# Short when 12h Williams %R > -20 (overbought) + price below 1d EMA50 + volume spike.
# Uses 1d EMA50 as trend filter to ensure trades align with higher timeframe trend.
# Designed for low trade frequency (10-25/year) to avoid fee drag in choppy markets.

name = "6h_WilliamsR_12h_1dEMA50_Volume"
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
    
    # Get 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R on 12h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 14-period Williams %R
    period = 14
    highest_high = pd.Series(high_12h).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_12h).rolling(window=period, min_periods=period).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * (highest_high - close_12h) / (highest_high - lowest_low), 
                          -50)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h and 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 6h volume spike (1.5x 50-period EMA)
    vol_ema = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_spike = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) + price above 1d EMA50 + volume spike
            if williams_r_aligned[i] < -80 and close[i] > ema_50_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20) + price below 1d EMA50 + volume spike
            elif williams_r_aligned[i] > -20 and close[i] < ema_50_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or price breaks below EMA50
            if williams_r_aligned[i] > -50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or price breaks above EMA50
            if williams_r_aligned[i] < -50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals