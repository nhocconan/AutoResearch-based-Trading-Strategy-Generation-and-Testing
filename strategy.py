#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
Hypothesis: Trade breakouts of Camarilla R1/S1 levels derived from daily pivot points with volume confirmation and ATR-based trend filter. In both bull and bear markets, price tends to respect these institutional levels. Enter long when price breaks above R1 with volume > 1.5x average and ATR(14) rising (indicating momentum). Enter short when price breaks below S1 with volume confirmation and rising ATR. Uses tight conditions to limit trades to 12-37/year. ATR filter ensures we only trade in momentum regimes, reducing whipsaw in sideways markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1, S1
    R1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    S1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Calculate ATR(14) on daily
    atr_period = 14
    tr = np.maximum(high_1d - low_1d, np.maximum(abs(high_1d - np.roll(close_1d, 1)), abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # first TR
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Align Camarilla levels and ATR to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    atr_12h = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, atr_period)  # ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # ATR rising condition: current ATR > previous ATR
        atr_rising = i > 0 and not np.isnan(atr_12h[i-1]) and atr_12h[i] > atr_12h[i-1]
        
        if position == 0:
            # Long: price breaks above R1 + volume + ATR rising
            if close[i] > R1_12h[i] and vol_confirm and atr_rising:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + ATR rising
            elif close[i] < S1_12h[i] and vol_confirm and atr_rising:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or ATR stops rising
            if close[i] < S1_12h[i] or (i > 0 and not np.isnan(atr_12h[i-1]) and atr_12h[i] <= atr_12h[i-1]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or ATR stops rising
            if close[i] > R1_12h[i] or (i > 0 and not np.isnan(atr_12h[i-1]) and atr_12h[i] <= atr_12h[i-1]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0