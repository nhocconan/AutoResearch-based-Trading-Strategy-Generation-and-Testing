#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R extreme with 1w EMA50 trend filter and volume confirmation.
- Williams %R(14) identifies overbought/oversold conditions: long when < -80, short when > -20
- 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades
- Volume confirmation (> 1.5x 20-period average) reduces false signals in low-participation moves
- Exit: Williams %R returns to neutral zone (-50) or opposite extreme
- Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag on 1d timeframe
- Williams %R works in both bull/bear markets by capturing mean reversion from extremes
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
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R(14) on 1d data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    williams_r = np.where(hl_range != 0, ((highest_high - close_1d) / hl_range) * -100, -50)
    
    # Align Williams %R to 1d timeframe (no additional delay needed for %R)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume MA, 14 for Williams %R, 50 for 1w EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: Williams %R < -80 (oversold) + volume spike + price > 1w EMA50 (uptrend)
            if volume_spike and close[i] > ema_50_aligned[i]:
                if williams_r_aligned[i] < -80:
                    signals[i] = 0.25
                    position = 1
            # Short entry: Williams %R > -20 (overbought) + volume spike + price < 1w EMA50 (downtrend)
            elif volume_spike and close[i] < ema_50_aligned[i]:
                if williams_r_aligned[i] > -20:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns to neutral (-50) or reaches overbought (> -20)
            if williams_r_aligned[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to neutral (-50) or reaches oversold (< -80)
            if williams_r_aligned[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_Extreme_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0