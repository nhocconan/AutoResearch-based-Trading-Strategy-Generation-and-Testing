#!/usr/bin/env python3
"""
1D_Camarilla_Pivot_Bounce_1wTrend_Volume
Hypothesis: Daily bounces off weekly Camarilla S3/R3 levels with volume confirmation and weekly trend filter. 
This strategy aims to capture mean-reversion bounces in ranging markets and continuation in trending markets 
by combining weekly trend alignment with daily price action at key weekly support/resistance levels. 
Designed for low trade frequency (<25/year) to minimize fee drift and work in both bull and bear markets.
"""

name = "1D_Camarilla_Pivot_Bounce_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Use previous weekly bar to calculate Camarilla levels (non-lookahad)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly range and Camarilla levels
    range_1w = high_1w - low_1w
    s3 = close_1w - (range_1w * 1.5)  # S3 level
    r3 = close_1w + (range_1w * 1.5)   # R3 level
    
    # Align weekly levels to daily timeframe (wait for weekly bar close)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    
    # Weekly trend filter: EMA 34 on weekly close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: volume > 2.0x 20-day average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction
        is_uptrend = close[i] > ema_34_1w_aligned[i]
        is_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price touches/bounces off S3 + volume + weekly uptrend
            if (low[i] <= s3_aligned[i] and 
                close[i] > s3_aligned[i] and  # Confirmed bounce
                volume[i] > vol_threshold[i] and 
                is_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: Price touches/rejects at R3 + volume + weekly downtrend
            elif (high[i] >= r3_aligned[i] and 
                  close[i] < r3_aligned[i] and  # Confirmed rejection
                  volume[i] > vol_threshold[i] and 
                  is_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price reaches opposite R3 level or loses weekly uptrend
            if high[i] >= r3_aligned[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price reaches opposite S3 level or loses weekly downtrend
            if low[i] <= s3_aligned[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals