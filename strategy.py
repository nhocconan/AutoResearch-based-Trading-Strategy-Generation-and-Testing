#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Buy breakouts above R1 (1w) in uptrend, sell breakdowns below S1 (1w) in downtrend, confirmed by volume spike. Weekly trend filter ensures alignment with major market direction. Volume surge confirms institutional participation. Designed for 12h timeframe to limit trades (target: 50-150 over 4 years) and reduce fee drag. Works in bull markets (buy R1 breakouts) and bear markets (sell S1 breakdowns).
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
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
    
    # === 1W Data for Camarilla Pivots and Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's OHLC for Camarilla calculation
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    # Camarilla levels: R1/S1 = C ± (H-L) * 1.1/12
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.1 / 12
    s1 = prev_close - rang * 1.1 / 12
    
    # Trend filter: EMA34 on 1w close (more responsive than EMA50)
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1W indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
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
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND uptrend (price > EMA34) AND volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_34_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND downtrend (price < EMA34) AND volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_34_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA34 OR reverses below R1
            if close[i] < ema_34_aligned[i] or close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above EMA34 OR reverses above S1
            if close[i] > ema_34_aligned[i] or close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals