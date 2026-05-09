#!/usr/bin/env python3
# Hypothesis: 12h Donchian breakout with 1d EMA100 trend filter and volume spike
# Long when price breaks above upper Donchian(20) with EMA100 uptrend and volume > 1.8x average
# Short when price breaks below lower Donchian(20) with EMA100 downtrend and volume > 1.8x average
# Exit when price crosses the EMA100 in opposite direction or touches opposite Donchian band
# Uses Donchian for breakout structure, EMA for trend filter, volume for conviction
# Designed to capture strong directional moves with low trade frequency to minimize fee drag
# Target: 50-120 total trades over 4 years (12-30/year) with size 0.25

name = "12h_Donchian_20_1dEMA100_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA100 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    # Using rolling window for upper and lower bands
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.8 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for EMA100 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema100_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, EMA100 uptrend, volume spike
            if (close[i] > donchian_upper[i] and 
                ema100_1d_aligned[i] > ema100_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, EMA100 downtrend, volume spike
            elif (close[i] < donchian_lower[i] and 
                  ema100_1d_aligned[i] < ema100_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA100 or touches lower Donchian
            if (close[i] < ema100_1d_aligned[i]) or (close[i] <= donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA100 or touches upper Donchian
            if (close[i] > ema100_1d_aligned[i]) or (close[i] >= donchian_upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals