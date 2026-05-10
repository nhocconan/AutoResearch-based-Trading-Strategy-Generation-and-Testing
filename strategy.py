#!/usr/bin/env python3
# 4h_KAMA_Trend_With_Volume_Confirm
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a dynamic trend filter.
# In trending markets, KAMA closely follows price; in ranging markets, it stays flat.
# Strategy: Go long when price crosses above KAMA with volume confirmation, short when price crosses below KAMA with volume confirmation.
# Uses daily trend filter (EMA34) to avoid counter-trend trades. Volume confirmation requires 2.0x 20-period MA to filter noise.
# Designed for low trade frequency (<50/year) to minimize fee drag and work in both bull and bear markets.

name = "4h_KAMA_Trend_With_Volume_Confirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle volatility calculation for array
    volatility_full = np.zeros_like(close)
    for i in range(er_len, len(close)):
        volatility_full[i] = np.sum(np.abs(np.diff(close[i-er_len:i])))
    er = np.where(volatility_full != 0, change / volatility_full, 0)
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (er_len), EMA34 (34), volume MA (20)
    start_idx = max(er_len, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (stricter: >2.0x MA to reduce false signals)
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: price crosses above KAMA + uptrend + volume
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below KAMA + downtrend + volume
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or trend breaks
            if close[i] < kama[i] or close[i-1] >= kama[i-1] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or trend breaks
            if close[i] > kama[i] or close[i-1] <= kama[i-1] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals