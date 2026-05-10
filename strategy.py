#!/usr/bin/env python3
"""
12h_KAMA_Trend_Filter_Volume
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a dynamic trend filter. Combined with 1d trend filter and volume confirmation on 12h timeframe, it captures sustained moves while avoiding whipsaws in both bull and bear markets. The 12h timeframe reduces trade frequency to minimize fee drag, and the adaptive nature of KAMA helps in ranging markets.
"""

name = "12h_KAMA_Trend_Filter_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 12h data for KAMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Parameters: ER period=10, Fast EMA=2, Slow EMA=30
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    vol = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0)  # This needs correction
    
    # Correct calculation of volatility (sum of absolute changes over ER period)
    er_period = 10
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    # Create a temporary array for volatility calculation
    temp_arr = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    vol = np.zeros_like(close_12h)
    for i in range(er_period, len(close_12h)):
        vol[i] = np.sum(temp_arr[i-er_period+1:i+1])
    # For initial periods, use expanding sum
    for i in range(er_period):
        vol[i] = np.sum(temp_arr[0:i+1])
    
    # Avoid division by zero
    er = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if vol[i] > 0:
            er[i] = change[i] / vol[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe (already on 12h, but we need to align to lower timeframe if needed)
    # Since we're using 12h as primary timeframe, we need to align 1d data to 12h
    # But KAMA is already calculated on 12h data, so we need to expand it to match the length of prices
    # However, our prices are at 12h timeframe, so we can use kama directly if lengths match
    
    # Actually, we need to handle the fact that our main loop is on the prices dataframe
    # which is at 12h timeframe. So we need to align our 1d EMA to 12h prices
    
    # Let's restart with clearer approach
    
    # Get the actual prices array (should be 12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Recalculate everything properly aligned
    
    # 1d EMA for trend filter (already aligned above)
    
    # For KAMA, we'll calculate it on the close prices directly since we're on 12h timeframe
    change = np.abs(np.diff(close, prepend=close[0]))
    temp_arr = np.abs(np.diff(close, prepend=close[0]))
    vol = np.zeros_like(close)
    for i in range(er_period, len(close)):
        vol[i] = np.sum(temp_arr[i-er_period+1:i+1])
    for i in range(er_period):
        vol[i] = np.sum(temp_arr[0:i+1])
    
    er = np.zeros_like(close)
    for i in range(len(close)):
        if vol[i] > 0:
            er[i] = change[i] / vol[i]
        else:
            er[i] = 0
    
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: current volume > 1.5x 20-period EMA of volume
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = max(34, er_period)  # Need 1d EMA34 and KAMA warmup
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(kama[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filters
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        uptrend_1d = close[i] > ema34_1d_aligned[i]
        downtrend_1d = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long entry: price above KAMA and uptrend on 1d with volume
            if price_above_kama and uptrend_1d and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA and downtrend on 1d with volume
            elif price_below_kama and downtrend_1d and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or 1d trend fails
            if not price_above_kama or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or 1d trend fails
            if not price_below_kama or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals