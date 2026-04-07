#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v2
Hypothesis: Weekly Camarilla pivot levels (S3/R3 for mean reversion, S4/R4 for breakout)
combined with daily EMA trend filter and volume confirmation works on 12h timeframe.
In ranging markets (weekly), fade at S3/R3 with daily EMA filter; in trending markets (weekly),
breakout at S4/R4. Volume confirmation reduces false signals. Targets 12-37 trades/year (50-150 over 4 years).
Works in both bull and bear markets by adapting to regime via weekly trend and daily EMA filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v2"
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
    
    # Weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly OHLC for pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Camarilla pivot levels
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    camarilla_r4 = close_1w + (high_1w - low_1w) * 1.1 / 2
    camarilla_r3 = close_1w + (high_1w - low_1w) * 1.1 / 4
    camarilla_s3 = close_1w - (high_1w - low_1w) * 1.1 / 4
    camarilla_s4 = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Daily data for EMA trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Daily volume for confirmation
    volume_1d = df_1d['volume'].values
    
    # Align weekly levels to 12h timeframe
    r4_12h = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    r3_12h = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    s4_12h = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Align daily indicators to 12h timeframe
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_12h = align_htf_to_ltf(prices, df_1d, 
                                  pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(r4_12h[i]) or np.isnan(s4_12h[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(vol_avg_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_12h[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 (mean reversion fail) OR 
            # price breaks above R4 and daily EMA turns down (breakout fail)
            if close[i] < s3_12h[i] or (close[i] > r4_12h[i] and close[i] < ema50_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (mean reversion fail) OR
            # price breaks below S4 and daily EMA turns up (breakout fail)
            if close[i] > r3_12h[i] or (close[i] < s4_12h[i] and close[i] > ema50_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion longs at S3 in uptrend (price > daily EMA)
            if (close[i] <= s3_12h[i] and 
                vol_confirm and 
                close[i] > ema50_12h[i]):
                position = 1
                signals[i] = 0.25
            # Mean reversion shorts at R3 in downtrend (price < daily EMA)
            elif (close[i] >= r3_12h[i] and 
                  vol_confirm and 
                  close[i] < ema50_12h[i]):
                position = -1
                signals[i] = -0.25
            # Breakout longs at R4 in uptrend
            elif (close[i] >= r4_12h[i] and 
                  vol_confirm and 
                  close[i] > ema50_12h[i]):
                position = 1
                signals[i] = 0.25
            # Breakout shorts at S4 in downtrend
            elif (close[i] <= s4_12h[i] and 
                  vol_confirm and 
                  close[i] < ema50_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals