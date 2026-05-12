#!/usr/bin/env python3
"""
12H_CAMARILLA_R3_S3_BREAKOUT_1DVOLUMESPIKE
Hypothesis: Daily Camarilla R3/S3 breakout on 12h chart with volume confirmation.
Trades breakouts from daily volatility-based levels (R3/S3) with volume spike filter.
Uses 12h timeframe to limit trade frequency (target: 50-150 total over 4 years).
Volume spike (2.5x 30-period average) confirms institutional participation.
No additional filters to keep edge pure; relies on volatility expansion + volume.
Works in bull markets (breakouts continue with trend) and bear markets (mean reversion from extremes).
"""
name = "12H_CAMARILLA_R3_S3_BREAKOUT_1DVOLUMESPIKE"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: volume > 2.5 * 30-period average (adjusted for 12h)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    rang = prev_high - prev_low
    R3 = prev_close + rang * 1.1 / 2
    S3 = prev_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup for volume MA
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume spike
            if close[i] > R3_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike
            elif close[i] < S3_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters the range (between S3 and R3)
            if close[i] < R3_aligned[i] and close[i] > S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters the range (between S3 and R3)
            if close[i] < R3_aligned[i] and close[i] > S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals