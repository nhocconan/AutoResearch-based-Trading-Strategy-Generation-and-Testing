#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1d ATR filter and volume spike
# Long when price breaks above Donchian upper band, ATR rising, volume > 2x average
# Short when price breaks below Donchian lower band, ATR rising, volume > 2x average
# Exit when price crosses the Donchian middle band
# Uses Donchian channels for breakout signals, ATR for volatility confirmation, volume for conviction
# Designed to capture breakouts with controlled frequency in both trending and ranging markets
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "4h_Donchian_Breakout_1dATR_VolumeSpike"
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
    
    # Calculate 1d Donchian channels (20-period high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's high/low for Donchian calculation
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    # Calculate Donchian upper and lower bands
    upper = prev_high.rolling(window=20, min_periods=20).max()
    lower = prev_low.rolling(window=20, min_periods=20).min()
    middle = (upper + lower) / 2
    
    # Align Donchian bands to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper.values)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower.values)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle.values)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean()
    
    # Align ATR to 4h timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14.values)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ATR calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or
            np.isnan(atr14_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper band, ATR rising, volume spike
            if (close[i] > upper_aligned[i] and 
                atr14_aligned[i] > atr14_aligned[i-1] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band, ATR rising, volume spike
            elif (close[i] < lower_aligned[i] and 
                  atr14_aligned[i] > atr14_aligned[i-1] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle band
            if close[i] < middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle band
            if close[i] > middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals