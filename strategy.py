#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_Volume_Spike
Strategy: 1d Donchian(20) breakout with weekly trend filter and volume spike.
Long: Price breaks above 20-day high + weekly trend up + volume > 2x average
Short: Price breaks below 20-day low + weekly trend down + volume > 2x average
Exit: Opposite Donchian breakout or volume drops below average
Position size: 0.25
Designed to capture strong trending moves with volume confirmation.
Timeframe: 1d
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
    
    # Calculate 20-day Donchian channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly trend using EMA34 on weekly data
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_series_1w = pd.Series(close_1w)
    ema34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike detection: current volume > 2x 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_roll[i-1]  # Break above previous period high
        breakout_down = close[i] < low_roll[i-1]  # Break below previous period low
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema34_1w_aligned[i]
        weekly_downtrend = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: upward breakout + weekly uptrend + volume spike
            if breakout_up and weekly_uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + weekly downtrend + volume spike
            elif breakout_down and weekly_downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: downward breakout or loss of volume spike
            if breakout_down or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: upward breakout or loss of volume spike
            if breakout_up or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_Volume_Spike"
timeframe = "1d"
leverage = 1.0