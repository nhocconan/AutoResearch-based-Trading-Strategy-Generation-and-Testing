#!/usr/bin/env python3
"""
4h Camarilla Pivot R1/S1 Breakout with Volume Confirmation
Hypothesis: Camarilla pivot levels act as strong support/resistance. Price breaking through R1/S1 with volume
indicates institutional breakout. Works in both bull (breakouts up) and bear (breakdowns down) by following
price action. Uses volume confirmation to avoid false breakouts and limits trades to reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_cci(high, low, close, period=20):
    """Calculate Commodity Channel Index"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tp = (high + low + close) / 3
    ma = np.zeros_like(tp)
    for i in range(len(tp)):
        if i < period:
            ma[i] = np.mean(tp[0:i+1])
        else:
            ma[i] = np.mean(tp[i-period+1:i+1])
    md = np.zeros_like(tp)
    for i in range(len(tp)):
        if i < period:
            md[i] = np.mean(np.abs(tp[0:i+1] - ma[i]))
        else:
            md[i] = np.mean(np.abs(tp[i-period+1:i+1] - ma[i]))
    cci = np.zeros_like(tp)
    for i in range(len(tp)):
        if md[i] != 0:
            cci[i] = (tp[i] - ma[i]) / (0.015 * md[i])
        else:
            cci[i] = 0
    return cci

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_r1[i] = close_1d[i]  # placeholder
            camarilla_s1[i] = close_1d[i]
        else:
            rang = high_1d[i-1] - low_1d[i-1]
            camarilla_r1[i] = close_1d[i-1] + rang * 1.1 / 12
            camarilla_s1[i] = close_1d[i-1] - rang * 1.1 / 12
    
    # Align to 4h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 25  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike
            if close[i] > camarilla_r1_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike
            elif close[i] < camarilla_s1_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below S1 (mean reversion) or volume dies
            if close[i] < camarilla_s1_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above R1 or volume dies
            if close[i] > camarilla_r1_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0