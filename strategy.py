#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla pivot S1/R1 breakout with 1-day volume spike and EMA34 trend filter.
Buy when price breaks above R1 with volume spike and rising EMA34; sell when price breaks below S1 with volume spike and falling EMA34.
Exit when price returns to the pivot point or volume dries up.
Camarilla levels provide institutional support/resistance; volume confirms institutional interest.
Designed for low trade frequency by requiring multiple confirmations (price, volume, trend).
Works in both bull and bear markets by following daily trend while using 4h Camarilla for precise entries.
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
    
    # Load 1-day data for EMA34 trend filter and pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily pivot points from previous day's OHLC
    # We need previous day's data to calculate today's Camarilla levels
    # Since we're using 4h data, we'll calculate pivots from the 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels for each day
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: S1 = C - (H-L)*1.08/2, R1 = C + (H-L)*1.08/2
    s1_1d = close_1d - range_1d * 1.08 / 2.0
    r1_1d = close_1d + range_1d * 1.08 / 2.0
    
    # Align these daily levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # Volume spike detection: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for volume MA
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1, volume spike, and EMA34 rising
            if (close[i] > r1_aligned[i] and 
                volume_spike[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, volume spike, and EMA34 falling
            elif (close[i] < s1_aligned[i] and 
                  volume_spike[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to pivot point OR volume dries up
                if (close[i] <= pivot_aligned[i] or 
                    volume[i] < vol_ma[i] * 0.5):  # Volume less than half of average
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to pivot point OR volume dries up
                if (close[i] >= pivot_aligned[i] or 
                    volume[i] < vol_ma[i] * 0.5):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0