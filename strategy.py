#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_Volume_Confirmation
Hypothesis: KAMA adapts to market noise - in trending markets it tracks price closely, in ranging markets it stays flat. 
Go long when price crosses above KAMA with volume confirmation (volume > 1.5x 20-day average), short when price crosses below KAMA with volume confirmation.
Uses weekly timeframe for trend filter to avoid counter-trend trades. Designed for 1d timeframe to limit trades (<20/year) and avoid fee drag.
Works in both bull (catches trend accelerations) and bear (catches trend reversals) markets.
"""

name = "1d_KAMA_Trend_Filter_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio (ER) for KAMA
    # ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.subtract(close_1d[10:], close_1d[:-10]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # temporary fix, will compute properly below
    
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        direction = np.abs(close_1d[i] - close_1d[i-10])
        volatility_sum = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
        if volatility_sum > 0:
            er[i] = direction / volatility_sum
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Get weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume average (20-day) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(kama_aligned[i-1]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: price crosses above KAMA + volume spike + price above weekly EMA50
            if close[i-1] <= kama_aligned[i-1] and close[i] > kama_aligned[i] and vol_spike and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price crosses below KAMA + volume spike + price below weekly EMA50
            elif close[i-1] >= kama_aligned[i-1] and close[i] < kama_aligned[i] and vol_spike and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA or price breaks below weekly EMA50
            if close[i] < kama_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA or price breaks above weekly EMA50
            if close[i] > kama_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals