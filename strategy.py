#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_Pullback
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
Enter long when KAMA turns up and price pulls back to KAMA with RSI < 40.
Enter short when KAMA turns down and price pulls back to KAMA with RSI > 60.
Exit on opposite KAMA cross.
Uses 1d ADX > 20 to ensure trending market, reducing false signals in chop.
Designed for low-frequency, high-conviction trades with controlled risk.
"""

name = "4h_KAMA_Trend_With_RSI_Pullback"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA (4h) - Adaptive moving average
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    change = np.abs(np.diff(close, n=1))
    change = np.concatenate([[0], change])
    
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    er = np.zeros(n)
    for i in range(er_period, n):
        if volatility[i] != 0:
            er[i] = change[i-er_period+1:i+1].sum() / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction (turning points)
    kama_up = kama > np.roll(kama, 1)
    kama_down = kama < np.roll(kama, 1)
    kama_up[0] = False
    kama_down[0] = False
    
    # RSI (4h) for pullback entries
    rsi_period = 14
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(1, n):
        if i < rsi_period:
            avg_gain[i] = np.mean(gain[1:i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[1:i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d ADX for trend filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder smoothing)
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr_period = 14
    atr = smooth_wilder(tr, atr_period)
    dm_plus_smooth = smooth_wilder(dm_plus, atr_period)
    dm_minus_smooth = smooth_wilder(dm_minus, atr_period)
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wilder(dx, atr_period)
    
    # Align indicators to 4h
    kama_up_aligned = align_htf_to_ltf(prices, df_1d, kama_up.astype(float))
    kama_down_aligned = align_htf_to_ltf(prices, df_1d, kama_down.astype(float))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_up_aligned[i]) or np.isnan(kama_down_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(close_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 20
        trending = adx_aligned[i] > 20
        
        if position == 0:
            # Long: KAMA turning up, price pulls back to KAMA, RSI < 40
            if (kama_up_aligned[i] and 
                close[i] <= kama[i] * 1.005 and  # Allow small overshoot
                rsi_aligned[i] < 40 and 
                trending):
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down, price pulls back to KAMA, RSI > 60
            elif (kama_down_aligned[i] and 
                  close[i] >= kama[i] * 0.995 and  # Allow small undershoot
                  rsi_aligned[i] > 60 and 
                  trending):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down
            if kama_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up
            if kama_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals