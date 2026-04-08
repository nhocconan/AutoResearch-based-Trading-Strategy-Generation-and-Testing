#!/usr/bin/env python3
"""
4h_12h1d_camarilla_pivot_v1
Hypothesis: Camarilla pivot levels on 1d chart with volume confirmation on 4h chart.
- Long when price touches S3 support with volume spike, short when touches R3 resistance with volume spike
- Use 12h trend filter (EMA25) to avoid counter-trend trades
- Designed for low trade frequency (20-40/year) to minimize fee drag
- Works in bull/bear via trend filter and mean reversion at extreme pivot levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h1d_camarilla_pivot_v1"
timeframe = "4h"
leverage = 1.0

def calculate_ema(values, period):
    """Calculate EMA with proper handling of NaN values"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    
    ema = np.full_like(values, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(values[:period])
    for i in range(period, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        range_val = ph - pl
        
        if range_val > 0:
            camarilla_r3[i] = pc + range_val * 1.1 / 2
            camarilla_s3[i] = pc - range_val * 1.1 / 2
            camarilla_r4[i] = pc + range_val * 1.1
            camarilla_s4[i] = pc - range_val * 1.1
    
    # Calculate 12h EMA (25-period) for trend filter
    close_12h = df_12h['close'].values
    ema_25_12h = calculate_ema(close_12h, 25)
    
    # Align indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # Volume confirmation: 20-period average on 4h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(25, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_25_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        trend_up = price > ema_25_12h_aligned[i]
        
        if position == 1:  # Long
            # Exit: price reaches S4 or trend turns down
            if price <= s4 or not trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price reaches R4 or trend turns up
            if price >= r4 or trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches S3 with volume spike and uptrend
            if abs(price - s3) < 0.001 * price and vol_ratio > 2.0 and trend_up:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches R3 with volume spike and downtrend
            elif abs(price - r3) < 0.001 * price and vol_ratio > 2.0 and not trend_up:
                position = -1
                signals[i] = -0.25
    
    return signals