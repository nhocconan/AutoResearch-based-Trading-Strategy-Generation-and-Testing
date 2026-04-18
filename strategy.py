#!/usr/bin/env python3
"""
1h_4h1d_MultiTimeframe_PivotBreakout_VolumeFilter
Hypothesis: Combine 4h and 1d timeframe structure with 1h entry timing. Use 4h EMA34 for trend direction, 1d Camarilla pivot levels for key support/resistance, and 1h volume spike for entry timing. Go long when price is above 4h EMA34, breaks above 1d S1 with volume confirmation; short when below 4h EMA34, breaks below 1d R1 with volume confirmation. Designed to capture momentum in both bull and bear markets by aligning with higher timeframe structure. Targets 15-30 trades/year with position size 0.20.
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
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34
    ema34_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 34:
        ema34_4h[33] = np.mean(close_4h[:34])
        for i in range(34, len(close_4h)):
            ema34_4h[i] = (close_4h[i] * 2/35) + (ema34_4h[i-1] * 33/35)
    
    # Align 4h EMA34 to 1h timeframe (wait for 4h bar close)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (S1, R1)
    # S1 = Close - 1.05 * (High - Low)
    # R1 = Close + 1.05 * (High - Low)
    camarilla_s1 = close_1d - 1.05 * (high_1d - low_1d)
    camarilla_r1 = close_1d + 1.05 * (high_1d - low_1d)
    
    # Align 1d Camarilla levels to 1h timeframe (wait for daily bar close)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    
    # Calculate 1h volume average (24-period = 1 day) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0 * 24-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long entry: above 4h EMA34, breaks above 1d S1 with volume confirmation
            if close[i] > ema34_4h_aligned[i] and close[i] > s1_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short entry: below 4h EMA34, breaks below 1d R1 with volume confirmation
            elif close[i] < ema34_4h_aligned[i] and close[i] < r1_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses back below 1d S1 or below 4h EMA34
            if close[i] < s1_1d_aligned[i] or close[i] < ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses back above 1d R1 or above 4h EMA34
            if close[i] > r1_1d_aligned[i] or close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_MultiTimeframe_PivotBreakout_VolumeFilter"
timeframe = "1h"
leverage = 1.0