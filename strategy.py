#!/usr/bin/env python3
name = "6h_ElderRay_2Bar_Trend_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 13-period EMA for Elder Ray (standard)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # 2-bar trend: Bull Power rising for 2 consecutive bars OR Bear Power falling for 2 consecutive bars
    bull_power_rising = (bull_power_1d[1:] > bull_power_1d[:-1]) & (bull_power_1d[2:] > bull_power_1d[1:-1])
    bear_power_falling = (bear_power_1d[1:] < bear_power_1d[:-1]) & (bear_power_1d[2:] < bear_power_1d[1:-1])
    
    # Pad to match original length
    bull_power_rising_2bar = np.concatenate([[False, False], bull_power_rising])
    bear_power_falling_2bar = np.concatenate([[False, False], bear_power_falling])
    
    # Trend filter: 50 EMA on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 6h timeframe
    bull_power_rising_2bar_aligned = align_htf_to_ltf(prices, df_1d, bull_power_rising_2bar)
    bear_power_falling_2bar_aligned = align_htf_to_ltf(prices, df_1d, bear_power_falling_2bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(bull_power_rising_2bar_aligned[i]) or np.isnan(bear_power_falling_2bar_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 2-bar bull power rising + price above EMA50 + volume
            if bull_power_rising_2bar_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: 2-bar bear power falling + price below EMA50 + volume
            elif bear_power_falling_2bar_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: trend reversal signal
            if position == 1:
                if bear_power_falling_2bar_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bull_power_rising_2bar_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals