#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Precision_v1
Hypothesis: Use Camarilla pivot levels from prior day for breakout entries on 4h chart.
Long when price breaks above R1 with volume > 1.8x 20-period average, short when breaks below S1.
Exit on opposite Camarilla level (S1 for longs, R1 for shorts) or when price returns to pivot point.
Uses tight volume confirmation (1.8x) to limit trades to ~25-40 per year. Camarilla levels work well
in ranging markets (common in 2025 BTC/ETH) and capture breakouts from key intraday levels.
Works in bull/bear by following breakout direction from statistically significant levels.
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
    
    # Get 1d data for Camarilla calculation (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day using prior day's OHLC
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # R2 = Close + 0.6 * (High - Low)
    # R1 = Close + 0.375 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    # S1 = Close - 0.375 * (High - Low)
    # S2 = Close - 0.6 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior day's data
    rng = high_1d - low_1d
    R1 = close_1d + 0.375 * rng
    S1 = close_1d - 0.375 * rng
    Pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 4h timeframe (use prior day's levels for current day)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    Pivot_aligned = align_htf_to_ltf(prices, df_1d, Pivot)
    
    # Volume confirmation: volume > 1.8x 20-period average (tight for low trade frequency)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1)  # Need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(Pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume
            if close[i] > R1_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume
            elif close[i] < S1_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Pivot or breaks below S1
            if close[i] <= Pivot_aligned[i] or close[i] < S1_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Pivot or breaks above R1
            if close[i] >= Pivot_aligned[i] or close[i] > R1_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Precision_v1"
timeframe = "4h"
leverage = 1.0