#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 12h Supertrend filter and 1d volume spike confirmation.
Enter long when 6h price closes above 12h Supertrend (uptrend) and 1d volume > 2.0x 20-period average.
Enter short when 6h price closes below 12h Supertrend (downtrend) and 1d volume > 2.0x 20-period average.
Exit when price crosses back over the 12h Supertrend or volume drops below 1.5x average.
Uses 12h Supertrend for trend direction (proven effective on 6h/12h) and 1d volume spikes for confirmation.
Volume spikes indicate institutional participation, reducing false breakouts.
Target: 50-150 total trades over 4 years (12-37/year). Position size: 0.25.
Works in bull markets (trend continuation) and bear markets (trend reversals on volume).
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
    
    # Get 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_12h, np.nan, dtype=float)
    direction = np.full_like(close_12h, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    # Start calculation after warmup period
    start_idx = atr_period
    for i in range(start_idx, len(close_12h)):
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(close_12h[i]):
            supertrend[i] = np.nan
            direction[i] = direction[i-1] if i > 0 else 1
            continue
            
        if i == start_idx:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if close_12h[i-1] > supertrend[i-1]:
                # Previous close was above previous Supertrend (uptrend)
                supertrend[i] = max(upper_band[i], supertrend[i-1])
                if close_12h[i] <= supertrend[i]:
                    # Trend change to downtrend
                    direction[i] = -1
                    supertrend[i] = lower_band[i]
                else:
                    direction[i] = 1
            else:
                # Previous close was below previous Supertrend (downtrend)
                supertrend[i] = min(lower_band[i], supertrend[i-1])
                if close_12h[i] >= supertrend[i]:
                    # Trend change to uptrend
                    direction[i] = 1
                    supertrend[i] = upper_band[i]
                else:
                    direction[i] = -1
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Volume filter: 2.0x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above Supertrend (uptrend) and volume spike
            if (close[i] > supertrend_aligned[i] and 
                direction_aligned[i] == 1 and 
                volume[i] > vol_ma_20_aligned[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price below Supertrend (downtrend) and volume spike
            elif (close[i] < supertrend_aligned[i] and 
                  direction_aligned[i] == -1 and 
                  volume[i] > vol_ma_20_aligned[i] * 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Supertrend or volume drops
            if close[i] < supertrend_aligned[i] or volume[i] < vol_ma_20_aligned[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Supertrend or volume drops
            if close[i] > supertrend_aligned[i] or volume[i] < vol_ma_20_aligned[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12hSupertrend_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0