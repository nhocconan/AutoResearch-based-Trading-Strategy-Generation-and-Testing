#!/usr/bin/env python3
"""
6h_1d1w_camarilla_pivot_v1
Hypothesis: Use 1d Camarilla pivot levels for reversal and 1w trend filter to avoid counter-trend trades on 6h chart.
- Long when price touches S1/S2 with 1w uptrend and rejection candle
- Short when price touches R1/R2 with 1w downtrend and rejection candle
- Uses Camarilla levels as support/resistance with mean reversion in ranging markets
- Trend filter ensures we only trade with the weekly trend
- Designed for low trade frequency (10-25/year) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d1w_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r1 = close + range_val * 1.1 / 12
    r2 = close + range_val * 1.1 / 6
    r3 = close + range_val * 1.1 / 4
    r4 = close + range_val * 1.1 / 2
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def is_rejection_candle(open_price, high, low, close, direction):
    """Check for rejection candle (pin bar)"""
    body = abs(close - open_price)
    wick_size = high - low
    if wick_size == 0:
        return False
    
    if direction == 'bullish':  # Long rejection - long lower wick
        lower_wick = min(open_price, close) - low
        upper_wick = high - max(open_price, close)
        return lower_wick > 2 * upper_wick and lower_wick > 0.6 * wick_size
    else:  # bearish: short rejection - long upper wick
        lower_wick = min(open_price, close) - low
        upper_wick = high - max(open_price, close)
        return upper_wick > 2 * lower_wick and upper_wick > 0.6 * wick_size

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pivot = np.full(len(close_1d), np.nan)
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_r2 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    camarilla_s2 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_pivot[i] = pivot
        camarilla_r1[i] = r1
        camarilla_r2[i] = r2
        camarilla_r3[i] = r3
        camarilla_r4[i] = r4
        camarilla_s1[i] = s1
        camarilla_s2[i] = s2
        camarilla_s3[i] = s3
        camarilla_s4[i] = s4
    
    # Calculate 1w EMA (50-period) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Align indicators to 6h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_r2_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_s2_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        open_val = open_price[i]
        pivot = camarilla_pivot_aligned[i]
        r1 = camarilla_r1_aligned[i]
        r2 = camarilla_r2_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s1 = camarilla_s1_aligned[i]
        s2 = camarilla_s2_aligned[i]
        s3 = camarilla_s3_aligned[i]
        trend_up = price > ema_50_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price reaches R1 or trend turns down
            if price >= r1 or not trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price reaches S1 or trend turns up
            if price <= s1 or trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches S1/S2 with bullish rejection and uptrend
            if ((abs(price - s1) < 0.001 * price or abs(price - s2) < 0.001 * price) and
                is_rejection_candle(open_val, high[i], low[i], close[i], 'bullish') and
                trend_up):
                position = 1
                signals[i] = 0.25
            # Enter short: price touches R1/R2 with bearish rejection and downtrend
            elif ((abs(price - r1) < 0.001 * price or abs(price - r2) < 0.001 * price) and
                  is_rejection_candle(open_val, high[i], low[i], close[i], 'bearish') and
                  not trend_up):
                position = -1
                signals[i] = -0.25
    
    return signals