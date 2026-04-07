#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: On 12-hour timeframe, use daily Camarilla pivot levels for mean reversion in ranging markets, with volume confirmation and 1-day EMA trend filter to avoid counter-trend trades. Enter long at S3 support in uptrend with volume spike, short at R3 resistance in downtrend with volume spike. Exit at opposite pivot level (S1/R1). Designed for low frequency (12-37 trades/year) to avoid fee drag while capturing mean reversion in chop and trend continuation in trends. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) by using 1-day trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1/6)
    # S2 = C - (Range * 1.1/4)
    # S3 = C - (Range * 1.1/2)
    # R1 = C + (Range * 1.1/6)
    # R2 = C + (Range * 1.1/4)
    # R3 = C + (Range * 1.1/2)
    pivot = (d_high + d_low + d_close) / 3.0
    rng = d_high - d_low
    s3 = d_close - (rng * 1.1 / 2.0)
    s1 = d_close - (rng * 1.1 / 6.0)
    r1 = d_close + (rng * 1.1 / 6.0)
    r3 = d_close + (rng * 1.1 / 2.0)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # 1-day EMA for trend filter (200-period)
    d_ema200 = pd.Series(d_close).ewm(span=200, adjust=False).mean().values
    d_ema200_aligned = align_htf_to_ltf(prices, df_1d, d_ema200)
    
    # Volume confirmation: current volume > 2.0x 24-period average (2 days worth)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):  # Start after volume average warmup
        # Skip if daily data not available
        if np.isnan(s3_aligned[i]) or np.isnan(d_ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs daily EMA200
        uptrend = close[i] > d_ema200_aligned[i]
        downtrend = close[i] < d_ema200_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 24-period average
        vol_confirm = volume[i] > 2.0 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price reaches or goes above S1 (first support level)
            if close[i] >= s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price reaches or goes below R1 (first resistance level)
            if close[i] <= r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price at or below S3 support in uptrend with volume confirmation
            long_entry = (close[i] <= s3_aligned[i]) and uptrend and vol_confirm
            # Short entry: price at or above R3 resistance in downtrend with volume confirmation
            short_entry = (close[i] >= r3_aligned[i]) and downtrend and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals