#!/usr/bin/env python3
name = "4h_12h_Supertrend_Pullback_Entry"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Supertrend calculation (ATR-based)
    atr_len = 10
    atr_mult = 3.0
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing
    atr = np.zeros_like(close)
    atr[atr_len-1] = np.mean(tr[:atr_len])
    for i in range(atr_len, len(tr)):
        atr[i] = (atr[i-1] * (atr_len - 1) + tr[i]) / atr_len
    
    # Calculate upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (atr_mult * atr)
    lower_band = hl2 - (atr_mult * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
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
    
    # Get 12h trend for higher timeframe filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    sma_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    trend_12h = close_12h > sma_12h
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(supertrend[i]) or np.isnan(trend_12h_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Supertrend uptrend + 12h uptrend + price pullback to Supertrend + volume
            if (direction[i] == 1 and trend_12h_aligned[i] and 
                close[i] <= supertrend[i] * 1.02 and  # Allow small pullback above ST
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend + 12h downtrend + price pullback to Supertrend + volume
            elif (direction[i] == -1 and not trend_12h_aligned[i] and 
                  close[i] >= supertrend[i] * 0.98 and  # Allow small pullback below ST
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Supertrend turns down OR price breaks below Supertrend
            if direction[i] == -1 or close[i] < supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Supertrend turns up OR price breaks above Supertrend
            if direction[i] == 1 or close[i] > supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals