#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_SwingFailure_Pullback_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Swing high/low detection (fractals) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Simple swing detection: high > previous 2 and next 2 highs
    swing_high = np.zeros(len(high_1d), dtype=bool)
    swing_low = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            swing_high[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            swing_low[i] = True
    
    # Extract swing values
    swing_high_vals = np.where(swing_high, high_1d, np.nan)
    swing_low_vals = np.where(swing_low, low_1d, np.nan)
    
    # Forward fill to get most recent swing levels
    swing_high_ff = pd.Series(swing_high_vals).ffill().values
    swing_low_ff = pd.Series(swing_low_vals).ffill().values
    
    # Align to 6h timeframe
    resistance_swing = align_htf_to_ltf(prices, df_1d, swing_high_ff, additional_delay_bars=1)
    support_swing = align_htf_to_ltf(prices, df_1d, swing_low_ff, additional_delay_bars=1)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current vs 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Pullback strength: how deep the pullback is from swing level
    # For longs: how close to support; for shorts: how close to resistance
    pullback_long = (close - support_swing) / (resistance_swing - support_swing)
    pullback_short = (resistance_swing - close) / (resistance_swing - support_swing)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(resistance_swing[i]) or np.isnan(support_swing[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(pullback_long[i]) or np.isnan(pullback_short[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if invalid swing structure
        if resistance_swing[i] <= support_swing[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: pullback to support with volume
            if (pullback_long[i] < 0.3 and  # pulled back to lower 30% of swing range
                vol_ratio[i] > 1.5):         # volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: pullback to resistance with volume
            elif (pullback_short[i] < 0.3 and  # pulled back to lower 30% of swing range
                  vol_ratio[i] > 1.5):         # volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: failed hold or reversal
            if (pullback_long[i] > 0.7 or   # moved back to upper 70% (failure)
                vol_ratio[i] < 0.8):        # volume drying up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: failed hold or reversal
            if (pullback_short[i] > 0.7 or  # moved back to upper 70% (failure)
                vol_ratio[i] < 0.8):        # volume drying up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals