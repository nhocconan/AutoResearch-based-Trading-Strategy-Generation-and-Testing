#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Uses 1d Donchian channels for breakout signals, filtered by 1w EMA50 trend to avoid counter-trend trades,
# and confirmed by 1d volume spike (1.5x 20-period average). Designed for 30-100 total trades over 4 years
# (7-25/year) to minimize fee drag. Works in bull markets via breakouts and in bear markets via
# trend-filtered short breakouts. All indicators calculated once before loop with proper alignment.

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and volume - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper channel = highest high over past 20 periods
    # Lower channel = lowest low over past 20 periods
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume spike confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ma * 1.5)
    
    # Align 1d indicators to 1d timeframe (already aligned, but shift for completed bar)
    donchian_upper_aligned = np.roll(high_roll, 1)
    donchian_lower_aligned = np.roll(low_roll, 1)
    volume_spike_aligned = np.roll(volume_spike, 1)
    donchian_upper_aligned[0] = np.nan
    donchian_lower_aligned[0] = np.nan
    volume_spike_aligned[0] = False
    
    # Get 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe (wait for completed 1w bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper channel AND volume spike AND price > 1w EMA50 (uptrend)
            if (close[i] > donchian_upper_aligned[i] and 
                volume_spike_aligned[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower channel AND volume spike AND price < 1w EMA50 (downtrend)
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_spike_aligned[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR volume drops
            if (close[i] <= donchian_upper_aligned[i] and close[i] >= donchian_lower_aligned[i]) or not volume_spike_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR volume drops
            if (close[i] <= donchian_upper_aligned[i] and close[i] >= donchian_lower_aligned[i]) or not volume_spike_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals