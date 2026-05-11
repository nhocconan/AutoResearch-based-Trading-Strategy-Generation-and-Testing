#!/usr/bin/env python3
"""
12h_Weekly_R1_S1_Breakout_WeeklyTrend_VolumeSpike
Hypothesis: Weekly R1/S1 breakouts with weekly trend filter (EMA50) and volume spike confirmation. Works in bull markets (buy R1 breakouts in uptrend) and bear markets (sell S1 breakdowns in downtrend). Volume spike confirms institutional interest. Target: 15-30 trades per year on 12h timeframe.
"""

name = "12h_Weekly_R1_S1_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "12h"
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
    
    # === Weekly Data for Weekly R1/S1 and Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's OHLC for weekly R1/S1 calculation
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = high_1w[0]  # First week uses same week's high
    prev_low[0] = low_1w[0]    # First week uses same week's low
    prev_close[0] = close_1w[0] # First week uses same week's close
    
    # Weekly R1/S1: C ± (H-L) * 1.1/12
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.1 / 12
    s1 = prev_close - rang * 1.1 / 12
    
    # Weekly trend filter: EMA50 on weekly close
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND uptrend (price > EMA50) AND volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND downtrend (price < EMA50) AND volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA50 OR reverses below R1
            if close[i] < ema_50_aligned[i] or close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above EMA50 OR reverses above S1
            if close[i] > ema_50_aligned[i] or close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals