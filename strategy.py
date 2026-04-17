#!/usr/bin/env python3
"""
12h_WAVES_1W_Volume_Signal_v1
Hypothesis: In both bull and bear markets, extreme weekly price action combined with volume spikes indicates institutional interest and potential trend continuation. 
We use 1-week high/low breakouts with volume confirmation on 12h timeframe to capture these moves. 
The strategy avoids counter-trend trades by using 1-week EMA as a trend filter. 
Target: 15-35 trades per year (60-140 total over 4 years) to stay within optimal frequency range.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1-week high/low for breakout levels ===
    df_1w = get_htf_data(prices, '1w')
    # Use weekly high and low as breakout levels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly levels to 12h timeframe (wait for weekly close)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # === 1-week EMA for trend filter ===
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Volume spike detector (20-period volume average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly high + volume spike + above weekly EMA50
            if (close[i] > weekly_high_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly low + volume spike + below weekly EMA50
            elif (close[i] < weekly_low_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below weekly low OR volume drops significantly
            if (close[i] < weekly_low_aligned[i] or 
                volume[i] < vol_ma_20[i] * 0.5):  # Volume drops below half average
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly high OR volume drops significantly
            if (close[i] > weekly_high_aligned[i] or 
                volume[i] < vol_ma_20[i] * 0.5):  # Volume drops below half average
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WAVES_1W_Volume_Signal_v1"
timeframe = "12h"
leverage = 1.0