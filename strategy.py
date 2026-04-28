#!/usr/bin/env python3
"""
4h_ElderRay_BullBear_Power_1dTrend_VolumeFilter
Hypothesis: Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation on 4h timeframe. 
Elder Ray measures bullish/bearish strength relative to EMA. Long when Bull Power > 0 + uptrend + volume surge.
Short when Bear Power < 0 + downtrend + volume surge. Targets 20-50 trades/year by requiring multiple confluence factors.
Works in bull markets via Bull Power strength and in bear markets via Bear Power strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d closes
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 13-period EMA for Elder Ray (standard setting)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA
    bear_power = low - ema_13   # Bear Power: Low - EMA
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: Bull Power > 0 + uptrend + volume surge
        long_entry = (bull_power[i] > 0 and 
                     trend_up[i] and 
                     volume_surge[i])
        
        # Short: Bear Power < 0 + downtrend + volume surge
        short_entry = (bear_power[i] < 0 and 
                      trend_down[i] and 
                      volume_surge[i])
        
        # Exit conditions: power crosses zero or trend changes
        long_exit = (bull_power[i] <= 0) or (not trend_up[i])
        short_exit = (bear_power[i] >= 0) or (not trend_down[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_ElderRay_BullBear_Power_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0