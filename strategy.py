#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour strategy using Elder Ray (Bull/Bear Power) from 1-day timeframe with volume confirmation.
# Elder Ray measures bullish/bearish power relative to EMA13. Strong bullish power (price > EMA13) with volume indicates
# institutional buying; strong bearish power indicates selling. Works in both bull and bear markets by following institutional flow.
# Uses 1-day EMA13 as the reference for power calculation. Volume filter (>1.5x 20-period average) confirms participation.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for 1-day
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe (wait for 1d bar to close)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: strong bullish power with volume (buying pressure)
        if (bull_power_1d_aligned[i] > 0 and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short entry: strong bearish power with volume (selling pressure)
        elif (bear_power_1d_aligned[i] < 0 and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: power fades or reverses
        elif position == 1 and bull_power_1d_aligned[i] <= 0:
            signals[i] = 0.0
            position = 0
        elif position == -1 and bear_power_1d_aligned[i] >= 0:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dEMA13_VolumeFilter"
timeframe = "6h"
leverage = 1.0