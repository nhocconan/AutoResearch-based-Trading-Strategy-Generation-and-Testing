#!/usr/bin/env python3
# 6h_1d_Supertrend_Follow_VolumeFilter
# Hypothesis: On 6h timeframe, follow Supertrend from 1d timeframe with volume confirmation.
# Supertrend identifies trend direction and provides dynamic support/resistance.
# Volume confirmation ensures moves are backed by participation.
# Targets 15-30 trades/year by requiring trend alignment and volume spike.
# Works in bull (follow uptrend) and bear (follow downtrend) markets.

name = "6h_1d_Supertrend_Follow_VolumeFilter"
timeframe = "6h"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Supertrend (ATR=10, multiplier=3.0)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing
    def atr_wilder(high, low, close, period):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = np.full_like(tr, np.nan)
        if len(tr) < period:
            return atr
        # First value: simple average
        atr[period-1] = np.nanmean(tr[1:period])
        # Wilder smoothing
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_val = atr_wilder(high_1d, low_1d, close_1d, 10)
    
    # Basic upper and lower bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3.0 * atr_val)
    lower_band = hl2 - (3.0 * atr_val)
    
    # Supertrend calculation
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1d)):
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(atr_val[i]):
            continue
            
        if close_1d[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_1d[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend and direction to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long when Supertrend indicates uptrend and volume confirms
            if direction_aligned[i] == 1 and close[i] > supertrend_aligned[i]:
                if volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short when Supertrend indicates downtrend and volume confirms
            elif direction_aligned[i] == -1 and close[i] < supertrend_aligned[i]:
                if volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long when Supertrend flips to downtrend
            if direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Supertrend flips to uptrend
            if direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals