#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_weekly_trend_volume_v1
Hypothesis: Camarilla pivot levels from daily chart combined with weekly trend filter and volume confirmation
provides high-probability reversal entries in ranging markets and breakout continuations in trending markets.
Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend) by trading with
weekly trend. Targets 12-37 trades/year by requiring daily Camarilla pivot touch + volume spike + weekly trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_weekly_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for each daily bar
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    #          S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    rang = high_1d - low_1d
    r4 = close_1d + rang * 1.1 / 2
    r3 = close_1d + rang * 1.1 / 4
    r2 = close_1d + rang * 1.1 / 6
    r1 = close_1d + rang * 1.1 / 12
    s1 = close_1d - rang * 1.1 / 12
    s2 = close_1d - rang * 1.1 / 6
    s3 = close_1d - rang * 1.1 / 4
    s4 = close_1d - rang * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_12h[i]) or 
            np.isnan(r4_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(r2_12h[i]) or np.isnan(r1_12h[i]) or
            np.isnan(s1_12h[i]) or np.isnan(s2_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(s4_12h[i]) or
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 OR trend turns down
            if close[i] < s3_12h[i] or close[i] < ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR trend turns up
            if close[i] > r3_12h[i] or close[i] > ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches S3/S4 + volume + uptrend (weekly)
            if ((abs(close[i] - s3_12h[i]) < 0.001 * close[i] or abs(close[i] - s4_12h[i]) < 0.001 * close[i]) and 
                vol_confirm and 
                close[i] > ema50_12h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches R3/R4 + volume + downtrend (weekly)
            elif ((abs(close[i] - r3_12h[i]) < 0.001 * close[i] or abs(close[i] - r4_12h[i]) < 0.001 * close[i]) and 
                  vol_confirm and 
                  close[i] < ema50_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals