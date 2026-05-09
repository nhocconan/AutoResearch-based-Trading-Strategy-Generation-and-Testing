#!/usr/bin/env python3
# Hypothesis: 4h 20-period Donchian breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above upper Donchian channel with 12h EMA50 uptrend and volume > 1.5x 20-period average
# Short when price breaks below lower Donchian channel with 12h EMA50 downtrend and volume > 1.5x 20-period average
# Exit when price crosses back below/above the middle of Donchian channel (20-period midpoint)
# Uses Donchian for breakout structure, EMA for trend alignment, volume for conviction
# Designed to capture strong momentum moves with controlled frequency to minimize fee drag
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "4h_Donchian_20_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 20-period Donchian channel
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max()
    lower_channel = low_series.rolling(window=20, min_periods=20).min()
    middle_channel = (upper_channel + lower_channel) / 2
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_channel.iloc[i]) or np.isnan(lower_channel.iloc[i]) or 
            np.isnan(middle_channel.iloc[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, EMA50 uptrend, volume spike
            if (close[i] > upper_channel.iloc[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, EMA50 downtrend, volume spike
            elif (close[i] < lower_channel.iloc[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle of Donchian channel
            if close[i] < middle_channel.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle of Donchian channel
            if close[i] > middle_channel.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals