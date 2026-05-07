#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_Pivot_Breakout_1wTrend_Volume
Hypothesis: Price breaking above/below weekly Camarilla pivot levels (R3/S3) on daily chart, aligned with weekly EMA10 trend and daily volume confirmation, captures strong momentum in both bull and bear markets. Low-frequency signals via daily timeframe and confluence of pivot breakout, trend, and volume.
"""
name = "1d_Weekly_Camarilla_Pivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly Camarilla pivot levels calculation
    # Based on previous week's high, low, close
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    pivot = (wh + wl + wc) / 3.0
    range_val = wh - wl
    
    # Camarilla levels
    R3 = pivot + (range_val * 1.1 / 2)
    S3 = pivot - (range_val * 1.1 / 2)
    
    # Align to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Weekly EMA10 for trend filter
    ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Daily volume filter: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure weekly data is available
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_10_1w_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3 + weekly uptrend + volume
            if close[i] > R3_aligned[i] and close[i] > ema_10_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3 + weekly downtrend + volume
            elif close[i] < S3_aligned[i] and close[i] < ema_10_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to weekly pivot level
            if position == 1:
                if close[i] <= pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals