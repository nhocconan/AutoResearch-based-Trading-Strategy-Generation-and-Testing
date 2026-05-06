#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Chaikin Money Flow (CMF) for institutional flow confirmation
# Long when price breaks above 1-day Donchian upper channel (20-period high) with CMF > 0.1
# Short when price breaks below 1-day Donchian lower channel (20-period low) with CMF < -0.1
# Uses institutional flow (CMF) to confirm breakouts, reducing false signals
# Target: 12-25 trades per year (48-100 over 4 years) with 0.25 position sizing

name = "12h_1dCMF_Donchian20_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Donchian Channel (20-period high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-period high and low for Donchian channels
    high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values
    low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    upper_donchian = align_htf_to_ltf(prices, df_1d, high_20)
    lower_donchian = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 1-day Chaikin Money Flow (20-period)
    # CMF = sum((Close - Low - (High - Close)) * Volume) / sum(Volume) over period
    mf_multiplier = ((df_1d['close'] - df_1d['low']) - (df_1d['high'] - df_1d['close'])) / (df_1d['high'] - df_1d['low'])
    # Handle division by zero when high == low
    mf_multiplier = mf_multiplier.fillna(0)
    mf_volume = mf_multiplier * df_1d['volume']
    cmf_20 = mf_volume.rolling(window=20, min_periods=20).sum() / df_1d['volume'].rolling(window=20, min_periods=20).sum()
    cmf_values = cmf_20.fillna(0).values
    
    # Align CMF to 12h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf_values)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian/CMF warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(cmf_aligned[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian with bullish CMF (> 0.1)
            if close[i] > upper_donchian[i] and cmf_aligned[i] > 0.1:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower Donchian with bearish CMF (< -0.1)
            elif close[i] < lower_donchian[i] and cmf_aligned[i] < -0.1:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian (support break)
            if close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian (resistance break)
            if close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals