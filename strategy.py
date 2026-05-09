#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R4/S4 breakout with 1d ADX trend filter and volume spike
# Long when price breaks above R4 with 1d ADX > 25 and volume > 1.5x average
# Short when price breaks below S4 with 1d ADX > 25 and volume > 1.5x average
# Exit when price returns to the central pivot (PP)
# Uses ADX for strong trend filter to avoid whipsaws, reducing trade frequency
# Target: 50-120 total trades over 4 years (12-30/year) with size 0.25

name = "4h_Camarilla_R4S4_Breakout_1dADX_VolumeSpike"
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
    
    # Calculate 1d ADX for trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = df_1d['high'] - df_1d['high'].shift(1)
    dm_minus = df_1d['low'].shift(1) - df_1d['low']
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    atr = tr.rolling(window=14, min_periods=14).mean()
    dm_plus_smooth = dm_plus.rolling(window=14, min_periods=14).mean()
    dm_minus_smooth = dm_minus.rolling(window=14, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate 1d OHLC for Camarilla levels
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    # Calculate pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    # Calculate Camarilla levels
    r4 = pp + (prev_high - prev_low) * 1.5000
    s4 = pp - (prev_high - prev_low) * 1.5000
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R4, ADX > 25 (strong trend), volume spike
            if (close[i] > r4_aligned[i] and 
                adx_1d_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S4, ADX > 25 (strong trend), volume spike
            elif (close[i] < s4_aligned[i] and 
                  adx_1d_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to central pivot
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to central pivot
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals