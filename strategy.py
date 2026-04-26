#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: On 6h timeframe, Camarilla R3/S3 breakouts with 1-week EMA34 trend filter and volume spikes capture high-probability continuation moves. R3/S3 levels act as strong support/resistance where breaks often lead to sustained moves. Weekly EMA34 ensures alignment with major trend, avoiding counter-trend breakouts. Volume spike confirms institutional participation. Designed to work in both bull (breakouts with trend) and bear (sharp reversals at extremes) markets by requiring volume and trend alignment. Target: 80-160 total trades over 4 years (20-40/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Load 1w data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous 1-day OHLC
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We only need R3 and S3 for breakout signals
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    
    # Calculate for each 6h bar using previous 1-day data
    for i in range(n):
        # Get the most recent completed 1-day candle
        # We need to find which 1d bar corresponds to time <= current 6h bar
        if i >= 4:  # Need at least 4 6h bars to have a prior day (24h/6h=4)
            day_idx = (i // 4) - 1  # Previous day's index in 1d data
            if day_idx >= 0 and day_idx < len(df_1d):
                H = df_1d['high'].iloc[day_idx]
                L = df_1d['low'].iloc[day_idx]
                C = df_1d['close'].iloc[day_idx]
                range_hl = H - L
                camarilla_R3[i] = C + (range_hl * 1.1 / 4)
                camarilla_S3[i] = C - (range_hl * 1.1 / 4)
    
    # Volume spike detection: volume > 2.0x 20-period EMA
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_R3[i]) or
            np.isnan(camarilla_S3[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1-week trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Long logic: price breaks above R3 with volume spike + in uptrend (continuation)
        if close[i] > camarilla_R3[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below S3 with volume spike + in downtrend (continuation)
        elif close[i] < camarilla_S3[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to Camarilla H-L midpoint or trend reverses
        elif position == 1 and (close[i] < (camarilla_R3[i] + camarilla_S3[i]) / 2 or downtrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > (camarilla_R3[i] + camarilla_S3[i]) / 2 or uptrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0