#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h with 12h Williams %R for overbought/oversold and 1d volume surge filter
# Williams %R identifies extreme price levels (oversold < -80, overbought > -20)
# Volume surge (>1.5x 20-period average) confirms conviction behind moves
# Works in bull markets (buy oversold bounces) and bear markets (sell overbought bounces)
# Low trade frequency due to dual confirmation requirement

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Williams %R
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Williams %R (14 periods)
    williams_length = 14
    highest_high = pd.Series(df_12h['high']).rolling(window=williams_length, min_periods=williams_length).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=williams_length, min_periods=williams_length).min().values
    williams_r = -100 * (highest_high - df_12h['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Load 1d data ONCE for volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume average (20 periods)
    vol_ma_length = 20
    vol_ma = pd.Series(df_1d['volume']).rolling(window=vol_ma_length, min_periods=vol_ma_length).mean().values
    
    # Align volume average to 4h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need enough for Williams %R and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_current = volume[i]
        
        # Volume surge condition: current volume > 1.5x 20-period average
        volume_surge = vol_current > 1.5 * vol_ma_aligned[i]
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        if position == 0:
            # Enter long: oversold + volume surge
            if oversold and volume_surge:
                position = 1
                signals[i] = position_size
            # Enter short: overbought + volume surge
            elif overbought and volume_surge:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or overbought (> -20)
            if williams_r_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or oversold (< -80)
            if williams_r_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12hWilliamsR_1dVolumeSurge_v1"
timeframe = "4h"
leverage = 1.0