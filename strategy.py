#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + Volume + 1-day Trend Filter.
Elder Ray measures bull/bear power via EMA(13). Long when bull power > 0 and rising, short when bear power < 0 and falling.
Volume confirms strength. 1-day EMA(50) trend filter ensures alignment with higher timeframe trend.
Targets 80-120 total trades over 4 years (20-30/year) to balance opportunity and cost.
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
    
    # Calculate EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Slope of Elder Ray (3-period change)
    bull_power_slope = pd.Series(bull_power).diff(3).values
    bear_power_slope = pd.Series(bear_power).diff(3).values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_strong = volume > (vol_ma_20 * 1.5)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1-day EMA(50) for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Trend: price above/below EMA50
    uptrend = close > ema50_1d_aligned
    downtrend = close < ema50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(bull_power_slope[i]) or 
            np.isnan(bear_power_slope[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: bull power positive AND rising + volume + uptrend
        long_entry = (bull_power[i] > 0) and (bull_power_slope[i] > 0) and vol_strong[i] and uptrend[i]
        # Short conditions: bear power negative AND falling + volume + downtrend
        short_entry = (bear_power[i] < 0) and (bear_power_slope[i] < 0) and vol_strong[i] and downtrend[i]
        
        # Exit when power fades (opposite condition)
        exit_long = position == 1 and (bull_power[i] <= 0 or bull_power_slope[i] <= 0)
        exit_short = position == -1 and (bear_power[i] >= 0 or bear_power_slope[i] >= 0)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_elder_ray_power_volume"
timeframe = "6h"
leverage = 1.0