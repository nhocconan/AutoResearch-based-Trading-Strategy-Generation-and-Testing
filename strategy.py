#!/usr/bin/env python3
"""
1d_1w_camarilla_pivot_volume_reversal_v1
Hypothesis: Mean reversion at weekly Camarilla pivot levels with volume confirmation.
- Long when price touches S1 support with volume spike in downtrend (weekly)
- Short when price touches R1 resistance with volume spike in uptrend (weekly)
- Uses weekly trend filter to avoid counter-trend trades
- Designed for low trade frequency (10-20/year) to minimize fee drag
- Works in bear via mean reversion at extreme levels, avoids whipsaw via trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_reversal_v1"
timeframe = "1d"
leverage = 1.0

def calculate_rma(series, period):
    """Calculate Wilder's RMA (same as EMA with alpha=1/period)"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    result = np.full_like(series, np.nan, dtype=float)
    alpha = 1.0 / period
    result[period-1] = np.mean(series[:period])
    for i in range(period, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    return result

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full_like(high, np.nan, dtype=float)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    
    # Camarilla levels
    r4 = close + range_ * 1.500
    r3 = close + range_ * 1.250
    r2 = close + range_ * 1.166
    r1 = close + range_ * 1.083
    s1 = close - range_ * 1.083
    s2 = close - range_ * 1.166
    s3 = close - range_ * 1.250
    s4 = close - range_ * 1.500
    
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (using previous day's HLC)
    # We need to shift by 1 to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize Camarilla arrays
    r1 = np.full(len(close_1d), np.nan)
    r2 = np.full(len(close_1d), np.nan)
    r3 = np.full(len(close_1d), np.nan)
    r4 = np.full(len(close_1d), np.nan)
    s1 = np.full(len(close_1d), np.nan)
    s2 = np.full(len(close_1d), np.nan)
    s3 = np.full(len(close_1d), np.nan)
    s4 = np.full(len(close_1d), np.nan)
    
    # Calculate for each day using previous day's data
    for i in range(1, len(close_1d)):
        r1[i], r2[i], r3[i], r4[i], s1[i], s2[i], s3[i], s4[i] = calculate_camarilla(
            high_1d[i-1], low_1d[i-1], close_1d[i-1]
        )
    
    # Calculate 1w EMA (20-period) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        alpha = 2.0 / (20 + 1)
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_20_1w[i-1]
    
    # Calculate 1d volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align indicators to 1d timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        trend_up = price > ema_20_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price reaches S2 or volume drops
            if price >= s2_aligned[i] or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price reaches R2 or volume drops
            if price <= r2_aligned[i] or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches S1 with volume spike in downtrend (weekly)
            if abs(price - s1_aligned[i]) < 0.001 * price and vol_ratio > 2.0 and not trend_up:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches R1 with volume spike in uptrend (weekly)
            elif abs(price - r1_aligned[i]) < 0.001 * price and vol_ratio > 2.0 and trend_up:
                position = -1
                signals[i] = -0.25
    
    return signals