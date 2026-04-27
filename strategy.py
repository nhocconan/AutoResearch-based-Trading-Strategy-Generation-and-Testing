#!/usr/bin/env python3
"""
Hypothesis: 6-hour RSI(14) with 1-week RSI(49) trend filter and 1-day volume confirmation.
In oversold conditions (RSI<30) with weekly uptrend: long.
In overbought conditions (RSI>70) with weekly downtrend: short.
Weekly RSI provides robust trend filter less prone to whipsaw than moving averages,
while daily volume confirms participation. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, length=14):
    """Relative Strength Index with proper smoothing"""
    if len(close) < length + 1:
        return np.full_like(close, np.nan, dtype=np.float64)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan, dtype=np.float64)
    avg_loss = np.full_like(close, np.nan, dtype=np.float64)
    
    # First average
    avg_gain[length] = np.mean(gain[:length])
    avg_loss[length] = np.mean(loss[:length])
    
    # Smooth subsequent values
    for i in range(length + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (length - 1) + gain[i-1]) / length
        avg_loss[i] = (avg_loss[i-1] * (length - 1) + loss[i-1]) / length
    
    rs = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(length, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = np.inf
    
    rsi = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(length, len(close)):
        if rs[i] == np.inf:
            rsi[i] = 100.0
        else:
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 49:
        return np.zeros(n)
    
    # Calculate weekly RSI49 for trend
    wk_close = df_1w['close'].values
    rsi_49_1w = calculate_rsi(wk_close, 49)
    rsi_49_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_49_1w)
    
    # Get daily volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 6h RSI14
    rsi_14_6h = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI14 (14), weekly RSI49 (49), volume MA (20)
    start_idx = max(14, 49, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_14_6h[i]) or np.isnan(rsi_49_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 6h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Current indicators
        rsi_now = rsi_14_6h[i]
        weekly_rsi = rsi_49_1w_aligned[i]
        
        # Volume filter: volume > 1.3x daily average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Trend filter: weekly RSI > 50 = uptrend, < 50 = downtrend
        weekly_uptrend = weekly_rsi > 50
        weekly_downtrend = weekly_rsi < 50
        
        if position == 0:
            # Oversold with weekly uptrend: long
            if rsi_now < 30 and weekly_uptrend and vol_filter:
                signals[i] = size
                position = 1
            # Overbought with weekly downtrend: short
            elif rsi_now > 70 and weekly_downtrend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: overbought or trend change
            if rsi_now > 70 or weekly_rsi < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: oversold or trend change
            if rsi_now < 30 or weekly_rsi > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI14_WeeklyRSI49_Volume"
timeframe = "6h"
leverage = 1.0