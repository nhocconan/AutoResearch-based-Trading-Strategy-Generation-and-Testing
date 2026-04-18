#!/usr/bin/env python3
"""
4h_HTF_Pivot_Trend_Breakout_V1
Hypothesis: Use 12h pivot points (R1/S1) for entry levels, 1d EMA34 for trend direction, and volume spike for confirmation. Long when price crosses above R1 in an uptrend with volume > 1.5x average, short when price crosses below S1 in a downtrend with volume > 1.5x average. Exit when price touches the opposite pivot level (S1 for long, R1 for short). Designed for 4h timeframe to capture meaningful moves while limiting trades to 20-50/year. Works in bull markets via trend-following longs and in bear via trend-following shorts.
"""

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
    
    # Get 12h data for pivot points (using daily high/low/close)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot_12h = np.full_like(close_12h, np.nan)
    r1_12h = np.full_like(close_12h, np.nan)
    s1_12h = np.full_like(close_12h, np.nan)
    
    for i in range(len(close_12h)):
        pivot_12h[i] = (high_12h[i] + low_12h[i] + close_12h[i]) / 3.0
        r1_12h[i] = 2 * pivot_12h[i] - low_12h[i]
        s1_12h[i] = 2 * pivot_12h[i] - high_12h[i]
    
    # Align pivot levels to 4h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])  # Simple average for first value
        alpha = 2.0 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    
    # Align EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, vol_period) + 1  # EMA34 and volume MA need warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA34: up if close > EMA, down if close < EMA
        # Use previous bar's close to avoid look-ahead
        prev_close = close[i-1] if i > 0 else close[0]
        trend_up = prev_close > ema34_1d_aligned[i]
        trend_down = prev_close < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0 and vol_confirm:
            # Long: price crosses above R1 in uptrend
            if trend_up and close[i] > r1_12h_aligned[i] and close[i-1] <= r1_12h_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below S1 in downtrend
            elif trend_down and close[i] < s1_12h_aligned[i] and close[i-1] >= s1_12h_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches S1 (opposite pivot level)
            if close[i] <= s1_12h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches R1 (opposite pivot level)
            if close[i] >= r1_12h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Pivot_Trend_Breakout_V1"
timeframe = "4h"
leverage = 1.0