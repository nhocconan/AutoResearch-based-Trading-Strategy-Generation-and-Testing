#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and 1d ATR Filter
Long: Price breaks above Donchian(20) upper band + volume > 1.5x 4h volume SMA(20) + 1d ATR(14) < 0.03 * price
Short: Price breaks below Donchian(20) lower band + volume > 1.5x 4h volume SMA(20) + 1d ATR(14) < 0.03 * price
Exit: Price crosses midline (average of upper/lower band) or ATR-based stop
Uses Donchian channels for breakout, volume confirmation, and low-volatility filter to avoid chop
Target: 20-30 trades/year per symbol (80-120 total over 4 years)
"""

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
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = np.zeros_like(tr)
    for i in range(1, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    # Set first 13 values to NaN (not enough data)
    atr_14[:13] = np.nan
    
    # Align 1d ATR to 4h
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate Donchian(20) on 4h data
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(n):
        start = max(0, i - 19)
        highest_high[i] = np.max(high[start:i+1])
        lowest_low[i] = np.min(low[start:i+1])
    
    # Calculate midline for exit
    midline = (highest_high + lowest_low) / 2
    
    # Calculate 4h volume SMA(20)
    vol_sma_4h = np.zeros(n)
    for i in range(n):
        start = max(0, i - 19)
        if i >= 19:
            vol_sma_4h[i] = np.mean(volume[start:i+1])
        else:
            vol_sma_4h[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(20, 20)  # need Donchian20 and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_sma_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_4h[i]
        atr_val = atr_14_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        mid = midline[i]
        
        # Volatility filter: only trade when 1d ATR < 3% of price (low volatility environment)
        vol_filter = atr_val < 0.03 * price
        
        if position == 0:
            # Long: Price breaks above upper band + volume > 1.5x SMA + low volatility
            if price > upper and close[i-1] <= upper and vol > 1.5 * vol_sma_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band + volume > 1.5x SMA + low volatility
            elif price < lower and close[i-1] >= lower and vol > 1.5 * vol_sma_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below midline
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above midline
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_1dATRFilter"
timeframe = "4h"
leverage = 1.0