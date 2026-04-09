#!/usr/bin/env python3
# 6h_cci_adx_pullback_v1
# Hypothesis: Combines Commodity Channel Index (CCI) pullbacks with ADX trend strength on 6h timeframe.
# Uses 1d ADX for trend regime filtering (ADX > 25 = trending) and 6h CCI for entry timing.
# Long when: 1d ADX > 25 (trending) AND 6h CCI crosses above -100 from below (pullback in uptrend).
# Short when: 1d ADX > 25 (trending) AND 6h CCI crosses below +100 from above (pullback in downtrend).
# Designed to capture trend continuations after pullbacks in both bull and bear markets.
# Target: 20-30 trades/year (80-120 total over 4 years) with strict trend+pullback criteria.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_adx_pullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1. 1d ADX for trend regime (calculated on daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align indices
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (14-period)
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(arr[1:period])  # skip index 0 (nan)
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 2. 6h CCI (20-period) for pullback entries
    # Typical Price
    tp = (high + low + close) / 3
    
    # Moving Average of TP
    ma_tp = np.full(n, np.nan)
    for i in range(20, n):
        ma_tp[i] = np.mean(tp[i-19:i+1])
    
    # Mean Deviation
    md = np.full(n, np.nan)
    for i in range(20, n):
        md[i] = np.mean(np.abs(tp[i-19:i+1] - ma_tp[i]))
    
    # CCI
    cci = np.full(n, np.nan)
    for i in range(20, n):
        if md[i] != 0:
            cci[i] = (tp[i] - ma_tp[i]) / (0.015 * md[i])
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(adx_aligned[i]) or np.isnan(cci[i]) or np.isnan(cci[i-1]):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d ADX > 25 indicates trending market
        is_trending = adx_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: CCI crosses below +100 (end of pullback/overbought)
            if cci[i] < 100 and cci[i-1] >= 100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above -100 (end of pullback/oversold)
            if cci[i] > -100 and cci[i-1] <= -100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: CCI crosses above -100 from below in trending market
            if is_trending and cci[i] > -100 and cci[i-1] <= -100:
                position = 1
                signals[i] = 0.25
            # Enter short: CCI crosses below +100 from above in trending market
            elif is_trending and cci[i] < 100 and cci[i-1] >= 100:
                position = -1
                signals[i] = -0.25
    
    return signals