#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
Long when price breaks above 20-bar high AND 1d ATR(14) > 1.5x 50-bar median ATR (high volatility regime) AND 4h volume > 1.5x 20-bar average volume.
Short when price breaks below 20-bar low AND same volatility and volume conditions.
Exit when price touches the 20-bar midpoint or opposite Donchian level.
Uses 1d for volatility regime filter, 4h for execution and volume confirmation.
Designed to capture breakouts in high volatility environments across bull and bear markets. Target: 25-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range and ATR(14)
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d 50-bar median ATR for volatility regime
    atr_median_50 = pd.Series(atr14).rolling(window=50, min_periods=50).median().values
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Calculate 4h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volatility indicators to 4h timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr_median_50_aligned = align_htf_to_ltf(prices, df_1d, atr_median_50)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or
            np.isnan(mid_20[i]) or
            np.isnan(atr14_aligned[i]) or
            np.isnan(atr_median_50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when ATR > 1.5x median ATR (high volatility)
        vol_regime = atr14_aligned[i] > 1.5 * atr_median_50_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_high = close[i] > high_20[i]
        breakout_low = close[i] < low_20[i]
        
        # Exit conditions: touch midpoint or opposite level
        touch_mid = abs(close[i] - mid_20[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < low_20[i]) or \
                         (position == -1 and close[i] > high_20[i])
        
        if position == 0:
            # Long: break above 20-bar high with volume confirmation and volatility regime
            if (breakout_high and volume_confirmed and vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: break below 20-bar low with volume confirmation and volatility regime
            elif (breakout_low and volume_confirmed and vol_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch midpoint or break below low
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch midpoint or break above high
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolRegime_Volume"
timeframe = "4h"
leverage = 1.0