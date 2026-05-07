#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeS
# Hypothesis: Uses weekly EMA50 trend filter on daily chart with Camarilla R1/S1 breakouts.
# Long when price > R1 and above weekly EMA50; short when price < S1 and below weekly EMA50.
# Volume confirmation reduces false breakouts. Weekly trend filter adapts to bull/bear markets.
# Target: 15-25 trades/year to minimize fee drag while capturing major trends.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeS"
timeframe = "1d"
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
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla pivot calculation (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels: R1, S1
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # Align R1 and S1 to daily timeframe (no alignment needed as df_1d is already daily)
    r1_aligned = r1  # Already on daily timeframe
    s1_aligned = s1  # Already on daily timeframe
    
    # Volume spike filter on daily (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > R1, above weekly EMA50 trend, volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < S1, below weekly EMA50 trend, volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < R1 or below weekly EMA50 trend
            if close[i] < r1_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > S1 or above weekly EMA50 trend
            if close[i] > s1_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals