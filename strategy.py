#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h Supertrend filter and volume confirmation.
# Long when price closes above Donchian upper band with 12h Supertrend uptrend and volume > 1.5x 20-bar average.
# Short when price closes below Donchian lower band with 12h Supertrend downtrend and volume > 1.5x average.
# Exit when price reverses and closes below/above the opposite Donchian band.
# Uses discrete position sizing 0.25. Target: 75-200 total trades over 4 years on 4h timeframe.
# 12h Supertrend ensures we only trade in the direction of the intermediate trend, reducing false breakouts in ranging markets.
# Volume confirmation ensures breakouts are supported by participation, increasing reliability.

name = "4h_Donchian20_12hSupertrend_Trend_VolumeConfirm_v1"
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
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    if n < lookback + 1:
        return np.zeros(n)
    
    upper_band = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower_band = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    if len(close_12h) < 10:
        return np.zeros(n)
    
    # Calculate ATR(10) for 12h Supertrend
    tr1 = pd.Series(high_12h).diff().abs()
    tr2 = (pd.Series(high_12h) - pd.Series(close_12h).shift(1)).abs()
    tr3 = (pd.Series(low_12h) - pd.Series(close_12h).shift(1)).abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Calculate Supertrend (10, 3.0) on 12h data
    hl2_12h = (high_12h + low_12h) / 2
    upper_band_12h = hl2_12h + 3.0 * atr_12h
    lower_band_12h = hl2_12h - 3.0 * atr_12h
    
    # Initialize Supertrend arrays
    supertrend_12h = np.zeros_like(close_12h)
    direction_12h = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    supertrend_12h[0] = hl2_12h[0]
    direction_12h[0] = 1
    
    # Calculate Supertrend iteratively
    for i in range(1, len(close_12h)):
        # Upper band
        if upper_band_12h[i] < supertrend_12h[i-1] or close_12h[i-1] > supertrend_12h[i-1]:
            upper_band_12h[i] = upper_band_12h[i]
        else:
            upper_band_12h[i] = supertrend_12h[i-1]
        
        # Lower band
        if lower_band_12h[i] > supertrend_12h[i-1] or close_12h[i-1] < supertrend_12h[i-1]:
            lower_band_12h[i] = lower_band_12h[i]
        else:
            lower_band_12h[i] = supertrend_12h[i-1]
        
        # Supertrend and direction
        if direction_12h[i-1] == 1:
            if close_12h[i] <= lower_band_12h[i]:
                direction_12h[i] = -1
                supertrend_12h[i] = upper_band_12h[i]
            else:
                direction_12h[i] = 1
                supertrend_12h[i] = lower_band_12h[i]
        else:
            if close_12h[i] >= upper_band_12h[i]:
                direction_12h[i] = 1
                supertrend_12h[i] = lower_band_12h[i]
            else:
                direction_12h[i] = -1
                supertrend_12h[i] = upper_band_12h[i]
    
    # Align 12h Supertrend direction to 4h timeframe (wait for 12h bar to close)
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(direction_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price closes above upper band with 12h Supertrend uptrend and volume > 1.5x average
            if (close[i] > upper_band[i] and 
                direction_12h_aligned[i] == 1 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below lower band with 12h Supertrend downtrend and volume > 1.5x average
            elif (close[i] < lower_band[i] and 
                  direction_12h_aligned[i] == -1 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below lower band (reversal signal)
            if close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above upper band (reversal signal)
            if close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals