#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: Camarilla pivot levels on 1w identify key support/resistance levels.
Long when price touches S3 level with volume confirmation and price above 1w EMA50 (uptrend).
Short when price touches R3 level with volume confirmation and price below 1w EMA50 (downtrend).
Uses 1w trend filter to avoid counter-trend trades. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for Camarilla pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous week
    # Using (H+L+C)/3 as pivot
    ph = df_1w['high'].values
    pl = df_1w['low'].values
    pc = df_1w['close'].values
    pivot = (ph + pl + pc) / 3.0
    range_ = ph - pl
    
    # Camarilla levels: S3 = pivot - 1.1*range/2, R3 = pivot + 1.1*range/2
    s3 = pivot - 1.1 * range_ / 2.0
    r3 = pivot + 1.1 * range_ / 2.0
    
    # 1w EMA50 for trend filter
    ema_50 = df_1w['close'].ewm(span=50, adjust=False).mean()
    
    # Align all 1w data to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50.values)
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves above S3 or breaks below EMA50
            if close[i] > s3_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price moves below R3 or breaks above EMA50
            if close[i] < r3_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches S3 level with volume and price above EMA50
            if (abs(close[i] - s3_aligned[i]) < 0.001 * close[i] and vol_confirm and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches R3 level with volume and price below EMA50
            elif (abs(close[i] - r3_aligned[i]) < 0.001 * close[i] and vol_confirm and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals