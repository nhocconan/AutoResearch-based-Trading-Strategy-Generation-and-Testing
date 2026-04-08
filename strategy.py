#!/usr/bin/env python3
"""
1d_1w_camarilla_pivot_volume_reversal_v1
Hypothesis: Mean reversion at weekly Camarilla pivot levels with volume confirmation.
- Long when price touches S3 level with volume spike in downtrend
- Short when price touches R3 level with volume spike in uptrend
- Uses 1w Camarilla levels for key support/resistance
- Volume filter to confirm institutional interest at levels
- Designed for low trade frequency (10-20/year) to minimize fee drag
- Works in ranging markets where price respects pivot levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_reversal_v1"
timeframe = "1d"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    if len(high) == 0:
        return np.array([]), np.array([])
    pivot = (high + low + close) / 3
    range_val = high - low
    r3 = close + range_val * 1.1 / 2
    s3 = close - range_val * 1.1 / 2
    return r3, s3

def calculate_rsi(close, period=14):
    """Calculate RSI"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels
    r3_1w, s3_1w = calculate_camarilla(high_1w, low_1w, close_1w)
    
    # Calculate 1w RSI for trend filter
    rsi_1w = calculate_rsi(close_1w, 14)
    
    # Calculate 20-period volume average for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align indicators to daily timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        r3 = r3_1w_aligned[i]
        s3 = s3_1w_aligned[i]
        rsi = rsi_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price moves back above S3 or RSI overbought
            if price > s3 or rsi > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price moves back below R3 or RSI oversold
            if price < r3 or rsi < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches S3 with volume spike in downtrend (RSI < 40)
            if abs(price - s3) < 0.001 * price and vol_ratio > 2.0 and rsi < 40:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches R3 with volume spike in uptrend (RSI > 60)
            elif abs(price - r3) < 0.001 * price and vol_ratio > 2.0 and rsi > 60:
                position = -1
                signals[i] = -0.25
    
    return signals