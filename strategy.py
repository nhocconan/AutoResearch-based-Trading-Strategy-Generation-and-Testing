#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme with 1d EMA50 trend filter and volume confirmation.
- Williams %R(14) below -90 = oversold (long), above -10 = overbought (short)
- Only take longs when price > 1d EMA50 (uptrend), shorts when price < 1d EMA50 (downtrend)
- Volume confirmation: > 1.3x 20-period average to filter low-participation moves
- Exit: Williams %R returns to -50 (mean reversion) or opposite extreme
- Uses 12h timeframe for lower frequency (target: 50-150 total trades over 4 years)
- Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.3x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Williams %R and EMA50
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R for each 1d bar
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe (available after 1d bar closes)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume MA, 14 for Williams %R, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.3x average)
        volume_spike = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R < -90 (oversold) + price > 1d EMA50 (uptrend) + volume spike
            if volume_spike and williams_r_aligned[i] < -90 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -10 (overbought) + price < 1d EMA50 (downtrend) + volume spike
            elif volume_spike and williams_r_aligned[i] > -10 and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) OR goes above -10 (overbought)
            if williams_r_aligned[i] >= -50 or williams_r_aligned[i] > -10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) OR goes below -90 (oversold)
            if williams_r_aligned[i] <= -50 or williams_r_aligned[i] < -90:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0