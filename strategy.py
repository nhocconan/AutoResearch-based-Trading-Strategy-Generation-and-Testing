#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Breakout_Volume_V1
Hypothesis: Price reacts at Camarilla pivot levels (R1/S1) calculated from 1d OHLC.
Enter long when price crosses above R1 with volume > 1.5x 20-period average and 12h EMA34 confirms uptrend.
Enter short when price crosses below S1 with volume confirmation and 12h EMA34 confirms downtrend.
Exit on opposite signal or when price touches the central pivot (P).
Uses 4h timeframe with 12h EMA trend filter and volume confirmation to reduce false breakouts.
Designed for 15-30 trades/year to minimize fee drag while capturing meaningful moves in both bull and bear markets.
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(34)
    ema_period = 34
    ema_12h = np.full_like(close_12h, np.nan)
    
    if len(close_12h) >= ema_period:
        ema_12h[ema_period-1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 / (ema_period + 1)) + (ema_12h[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (using previous day's data)
    # P = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    p_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        p_1d[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        r1_1d[i] = close_1d[i-1] + (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
        s1_1d[i] = close_1d[i-1] - (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    p_1d_aligned = align_htf_to_ltf(prices, df_1d, p_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(ema_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(p_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above R1 with volume confirmation and EMA uptrend
            if close[i] > r1_1d_aligned[i] and close[i-1] <= r1_1d_aligned[i-1] and \
               volume[i] > 1.5 * vol_ma[i] and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below S1 with volume confirmation and EMA downtrend
            elif close[i] < s1_1d_aligned[i] and close[i-1] >= s1_1d_aligned[i-1] and \
                 volume[i] > 1.5 * vol_ma[i] and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches central pivot P or reverses below S1
            if close[i] <= p_1d_aligned[i] or close[i] < s1_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches central pivot P or reverses above R1
            if close[i] >= p_1d_aligned[i] or close[i] > r1_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_Volume_V1"
timeframe = "4h"
leverage = 1.0