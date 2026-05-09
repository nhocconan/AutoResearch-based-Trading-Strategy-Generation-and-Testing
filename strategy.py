#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume spike
# Long when price breaks above Donchian upper channel (20-period high) with ADX > 25 and volume > 2x average
# Short when price breaks below Donchian lower channel (20-period low) with ADX > 25 and volume > 2x average
# Exit when price crosses the Donchian midline (10-period average of high/low) or reverses to opposite channel
# Uses Donchian channels for breakout signals, ADX for trend strength confirmation, volume for conviction
# Designed to capture strong breakouts in trending markets while avoiding false signals in ranging conditions
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "4h_Donchian_Breakout_1dADX_VolumeSpike"
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
    
    # Previous day's 20-period high and low for Donchian calculation
    prev_high_max = df_1d['high'].rolling(window=20, min_periods=20).max().shift(1)
    prev_low_min = df_1d['low'].rolling(window=20, min_periods=20).min().shift(1)
    
    # Calculate Donchian midline (average of upper and lower channel)
    donchian_mid = (prev_high_max + prev_low_min) / 2
    
    # Align Donchian levels to 4h timeframe
    upper_channel = align_htf_to_ltf(prices, df_1d, prev_high_max.values)
    lower_channel = align_htf_to_ltf(prices, df_1d, prev_low_min.values)
    midline = align_htf_to_ltf(prices, df_1d, donchian_mid.values)
    
    # Calculate 1d ADX for trend strength filter
    # Calculate True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = df_1d['high'] - df_1d['high'].shift(1)
    dm_minus = df_1d['low'].shift(1) - df_1d['low']
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Calculate smoothed TR and DM
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean()
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean()
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # Calculate DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(midline[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper channel, ADX > 25, volume spike
            if (close[i] > upper_channel[i] and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel, ADX > 25, volume spike
            elif (close[i] < lower_channel[i] and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses midline or reverses to lower channel
            if (close[i] <= midline[i]) or (close[i] < lower_channel[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses midline or reverses to upper channel
            if (close[i] >= midline[i]) or (close[i] > upper_channel[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals