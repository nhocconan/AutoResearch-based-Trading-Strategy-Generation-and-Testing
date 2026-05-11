#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: Use 4h Camarilla R1/S1 breakouts with 4h EMA20 trend filter and volume spike confirmation on 1h timeframe. 
Designed to capture institutional breakouts in both bull and bear markets with strict entry criteria to limit trades and avoid fee drift.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla calculation and EMA20 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous day's OHLC for Camarilla calculation (using 4h as base)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    # Handle first day
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    
    # Calculate Camarilla R1 and S1 levels
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.1 / 4
    s1 = prev_close - rang * 1.1 / 4
    
    # Calculate EMA20 on 4h close
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume spike: current volume > 2.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND uptrend (price > EMA20) AND volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_20_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND downtrend (price < EMA20) AND volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_20_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA20 OR reverses below R1
            if close[i] < ema_20_4h_aligned[i] or close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price crosses above EMA20 OR reverses above S1
            if close[i] > ema_20_4h_aligned[i] or close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals