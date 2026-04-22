#!/usr/bin/env python3
"""
Hypothesis: 6-hour Donchian Channel breakout with 1-day trend filter and volume confirmation.
Long when price breaks above Donchian upper channel (20-period) and 1d EMA50 is rising with volume spike.
Short when price breaks below Donchian lower channel (20-period) and 1d EMA50 is falling with volume spike.
Exit when price crosses the Donchian middle (10-period average of high/low) or 1d EMA50 reverses.
Designed for low trade frequency by requiring multiple confirmations (breakout + trend + volume).
Works in both bull and bear markets by following the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    # Upper = max(high, 20), Lower = min(low, 20), Middle = (upper + lower) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: break above upper channel, 1d EMA50 rising, volume spike
            if (close[i] > donchian_upper[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel, 1d EMA50 falling, volume spike
            elif (close[i] < donchian_lower[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses middle or 1d EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle or 1d EMA50 turns down
                if close[i] < donchian_middle[i] or ema50_1d_aligned[i] < ema50_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle or 1d EMA50 turns up
                if close[i] > donchian_middle[i] or ema50_1d_aligned[i] > ema50_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0