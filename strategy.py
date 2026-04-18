#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1S1_Breakout_WeeklyTrend_Filter
Hypothesis: Daily Camarilla R1/S1 breakouts filtered by weekly trend (close > weekly EMA20) with volume confirmation. Targets 25-35 trades/year per symbol, works in both bull (breakouts) and bear (mean reversion via opposite breakouts) by capturing institutional levels. Weekly trend filter reduces false signals in chop.
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
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
    
    # Align daily Camarilla levels to 1d timeframe (no shift needed as same TF)
    camarilla_r1_aligned = camarilla_r1  # Already on daily scale
    camarilla_s1_aligned = camarilla_s1
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.zeros_like(close_1w)
    ema_20_1w[:] = np.nan
    if len(close_1w) >= 20:
        k = 2 / (20 + 1)
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = close_1w[i] * k + ema_20_1w[i-1] * (1 - k)
    
    # Align weekly EMA20 to daily timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 20  # Warmup for weekly EMA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and above weekly EMA20 (uptrend)
            if close[i] > camarilla_r1_aligned[i] and vol_spike[i] and close[i] > ema_20_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below S1 with volume spike and below weekly EMA20 (downtrend)
            elif close[i] < camarilla_s1_aligned[i] and vol_spike[i] and close[i] < ema_20_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: minimum 3 days hold, then exit on mean reversion or trend change
            if bars_since_entry >= 3:
                if close[i] < camarilla_s1_aligned[i] or close[i] < ema_20_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25  # Hold during minimum period
        
        elif position == -1:
            # Exit: minimum 3 days hold, then exit on mean reversion or trend change
            if bars_since_entry >= 3:
                if close[i] > camarilla_r1_aligned[i] or close[i] > ema_20_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25  # Hold during minimum period
    
    return signals

name = "1d_1w_Camarilla_R1S1_Breakout_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0