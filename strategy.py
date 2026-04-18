#!/usr/bin/env python3
"""
12h_VWAP_Trend_With_1d_Camarilla_R1S1
Hypothesis: Price above/below VWAP indicates trend direction, while 1d Camarilla R1/S1 levels provide precise entry/exit zones. VWAP filters noise, Camarilla levels capture institutional interest, and the combination works in both bull (buy VWAP bounces at S1) and bear (sell VWAP rejections at R1). Low-frequency 12h timeframe reduces trade frequency and fee drag.
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
    
    # Calculate VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_r1[i] = close_1d[i]
            camarilla_s1[i] = close_1d[i]
        else:
            rang = high_1d[i-1] - low_1d[i-1]
            camarilla_r1[i] = close_1d[i-1] + rang * 1.1 / 12
            camarilla_s1[i] = close_1d[i-1] - rang * 1.1 / 12
    
    # Align to 12h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above VWAP and tests S1 support
            if close[i] > vwap[i] and close[i] <= camarilla_s1_aligned[i] * 1.005:  # within 0.5% of S1
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP and tests R1 resistance
            elif close[i] < vwap[i] and close[i] >= camarilla_r1_aligned[i] * 0.995:  # within 0.5% of R1
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below VWAP (trend change) or reaches R1
            if close[i] < vwap[i] or close[i] >= camarilla_r1_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above VWAP (trend change) or reaches S1
            if close[i] > vwap[i] or close[i] <= camarilla_s1_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VWAP_Trend_With_1d_Camarilla_R1S1"
timeframe = "12h"
leverage = 1.0