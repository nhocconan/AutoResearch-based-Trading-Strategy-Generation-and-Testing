#!/usr/bin/env python3
"""
Hypothesis: 6-hour Donchian(20) breakout with 1-week trend filter and volume confirmation.
Long when price breaks above 20-period high with rising 1-week EMA50 and volume spike.
Short when price breaks below 20-period low with falling 1-week EMA50 and volume spike.
Exit when price retests the midpoint of the Donchian channel.
Designed for low trade frequency by requiring multiple confirmations and using weekly trend.
Works in both bull and bear markets by following the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1-week EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for EMA50
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(mid_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above 20-period high with rising 1-week EMA50 and volume spike
            if (close[i] > high_20[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low with falling 1-week EMA50 and volume spike
            elif (close[i] < low_20[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price retests midpoint of Donchian channel
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midpoint
                if close[i] < mid_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above midpoint
                if close[i] > mid_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian_20_1wEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0