#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1d EMA100 trend filter and volume confirmation
# Long when price breaks above 20-period high with 1d EMA100 uptrend and volume > 1.8x average
# Short when price breaks below 20-period low with 1d EMA100 downtrend and volume > 1.8x average
# Exit when price retraces to midline of Donchian channel
# Uses Donchian for breakout structure, EMA for trend, volume for conviction
# Designed to capture strong moves in trending markets while avoiding choppy periods
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "4h_Donchian_Breakout_1dEMA100_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 1d EMA100 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Calculate Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.8 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for EMA and Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema100_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, EMA100 uptrend, volume spike
            if (close[i] > donchian_high[i] and 
                ema100_1d_aligned[i] > ema100_1d_aligned[i-1] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, EMA100 downtrend, volume spike
            elif (close[i] < donchian_low[i] and 
                  ema100_1d_aligned[i] < ema100_1d_aligned[i-1] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retraces to midline
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retraces to midline
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals