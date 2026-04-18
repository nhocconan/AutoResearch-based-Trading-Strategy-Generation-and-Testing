#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_Volume_ATRFilter
Hypothesis: On 6h timeframe, price breaking through daily Camarilla R1/S1 levels with volume confirmation and ATR-based volatility filter captures institutional breakouts while avoiding false signals in low-volatility chop. Works in bull (upside breakouts) and bear (downside breakdowns) by following price action direction. ATR filter ensures trades occur only when volatility is sufficient for meaningful moves, reducing whipsaw in ranging markets.
"""

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
    
    # Get daily data for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR(14) on daily timeframe for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Pad first element
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14) - simple moving average of TR
    atr_1d = np.full_like(close_1d, np.nan)
    for i in range(14, len(tr)):
        atr_1d[i] = np.nanmean(tr[i-13:i+1])
    
    # Calculate Camarilla levels from previous day
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        rang = high_1d[i-1] - low_1d[i-1]
        camarilla_r1[i] = close_1d[i-1] + rang * 1.1 / 12
        camarilla_s1[i] = close_1d[i-1] - rang * 1.1 / 12
    
    # Align all indicators to 6h timeframe (use previous day's values)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    # Fill beginning with available data
    for i in range(20):
        vol_ma[i] = np.mean(volume[0:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    # ATR filter: current ATR > 0.8 * 20-period average ATR (avoid low volatility)
    atr_ma = np.full_like(atr_1d_aligned, np.nan)
    for i in range(20, len(atr_1d_aligned)):
        if not np.isnan(atr_1d_aligned[i]):
            atr_ma[i] = np.nanmean(atr_1d_aligned[i-19:i+1])
    for i in range(20):
        if not np.isnan(atr_1d_aligned[i]):
            atr_ma[i] = np.nanmean(atr_1d_aligned[0:i+1])
    atr_filter = (~np.isnan(atr_1d_aligned)) & (~np.isnan(atr_ma)) & (atr_1d_aligned > atr_ma * 0.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and sufficient volatility
            if close[i] > camarilla_r1_aligned[i] and vol_spike[i] and atr_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and sufficient volatility
            elif close[i] < camarilla_s1_aligned[i] and vol_spike[i] and atr_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below S1 (mean reversion) or volatility drops
            if close[i] < camarilla_s1_aligned[i] or not atr_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above R1 or volatility drops
            if close[i] > camarilla_r1_aligned[i] or not atr_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0