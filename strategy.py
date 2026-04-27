#!/usr/bin/env python3
"""
1d Camarilla Pivot Reversal with 1w Trend Filter and Volume Spike.
Long when price touches S3 or S4 and reverses up + 1w trend up + volume spike.
Short when price touches R3 or R4 and reverses down + 1w trend down + volume spike.
Exit when price reaches opposite pivot level (S1 for longs, R1 for shorts) or trend reverses.
Designed for low frequency (8-18 trades/year) to minimize fee drift.
Uses Camarilla pivots for reversal zones and 1w EMA for trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.full_like(close_1w, np.nan)
    for i in range(33, len(close_1w)):
        ema_1w[i] = np.mean(close_1w[i-33:i+1])
    
    # Align 1w EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily Camarilla pivots
    camarilla_r4 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_r2 = np.full(n, np.nan)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_s2 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        pc = close[i-1]
        ph = high[i-1]
        pl = low[i-1]
        rng = ph - pl
        camarilla_r4[i] = pc + rng * 1.1 / 2
        camarilla_r3[i] = pc + rng * 1.1 / 4
        camarilla_r2[i] = pc + rng * 1.1 / 6
        camarilla_r1[i] = pc + rng * 1.1 / 12
        camarilla_s1[i] = pc - rng * 1.1 / 12
        camarilla_s2[i] = pc - rng * 1.1 / 6
        camarilla_s3[i] = pc - rng * 1.1 / 4
        camarilla_s4[i] = pc - rng * 1.1 / 2
    
    # Volume filter: volume > 1.3x average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need pivots (1) + volume MA (20) + 1w EMA (34)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_now = close[i]
        vol_now = volume[i]
        
        r4 = camarilla_r4[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        s4 = camarilla_s4[i]
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        trend_1w = ema_1w_aligned[i]
        
        # Volume filter: volume > 1.3x average
        vol_filter = vol_now > 1.3 * vol_ma_20[i]
        
        # Reversal detection: price touches level and closes back inside
        touched_r4 = high[i] >= r4 and close[i] < r4
        touched_r3 = high[i] >= r3 and close[i] < r3
        touched_s3 = low[i] <= s3 and close[i] > s3
        touched_s4 = low[i] <= s4 and close[i] > s4
        
        if position == 0:
            # Long: price touches S3/S4 and reverses up + 1w trend up + volume spike
            if ((touched_s3 or touched_s4) and 
                price_now > s1 and  # Already reversed above S1
                trend_1w > close[i-1] and  # Uptrend
                vol_filter):
                signals[i] = size
                position = 1
            # Short: price touches R3/R4 and reverses down + 1w trend down + volume spike
            elif ((touched_r3 or touched_r4) and 
                  price_now < r1 and  # Already reversed below R1
                  trend_1w < close[i-1] and  # Downtrend
                  vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches S1 (opposite) or 1w trend turns down
            if price_now <= s1 or trend_1w < close[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches R1 (opposite) or 1w trend turns up
            if price_now >= r1 or trend_1w > close[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_CamarillaPivotReversal_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0