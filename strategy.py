#!/usr/bin/env python3
# [24891] 6h_1d_camarilla_pivot_v1
# Hypothesis: 6-hour Camarilla pivot with 1-day trend filter. Long when price breaks above R4 with 1-day close > EMA50. Short when price breaks below S4 with 1-day close < EMA50. Exit when price crosses the opposite pivot level or 1-day trend reverses. Uses 1-day EMA50 to filter direction and Camarilla for precise entry/exit. Designed for 6h timeframe to capture swing moves with low trade frequency (~15-30/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-day data for pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Calculate daily Camarilla levels (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values  # Previous day close
    
    # Initialize arrays
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d_prev[i])):
            range_ = high_1d[i-1] - low_1d[i-1]
            camarilla_r4[i] = close_1d_prev[i] + range_ * 1.1 / 2
            camarilla_r3[i] = close_1d_prev[i] + range_ * 1.1 / 4
            camarilla_s3[i] = close_1d_prev[i] - range_ * 1.1 / 4
            camarilla_s4[i] = close_1d_prev[i] - range_ * 1.1 / 2
    
    # Align Camarilla levels and EMA50 to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        r4 = r4_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        trend_up = close_1d[-1] > ema_50 if len(close_1d) > 0 else False  # Current 1-day close vs EMA
        
        if position == 1:  # Long
            # Exit: price crosses below S3 or 1-day trend turns down
            if price < s3 or (len(close_1d) > 0 and close_1d[-1] < ema_50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above R3 or 1-day trend turns up
            if price > r3 or (len(close_1d) > 0 and close_1d[-1] > ema_50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above R4 with 1-day uptrend
            if price > r4 and len(close_1d) > 0 and close_1d[-1] > ema_50:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below S4 with 1-day downtrend
            elif price < s4 and len(close_1d) > 0 and close_1d[-1] < ema_50:
                position = -1
                signals[i] = -0.25
    
    return signals