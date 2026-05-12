#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
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
    
    # ===== 4h Trend Filter (HTF) =====
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA(34) for trend
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # ===== 4h Camarilla Pivot Levels (HTF) =====
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_prev = np.roll(close_4h, 1)
    close_4h_prev[0] = close_4h[0]  # first day uses same close
    
    pivot = (high_4h + low_4h + close_4h_prev) / 3.0
    range_4h = high_4h - low_4h
    
    # Camarilla levels: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    r1 = close_4h_prev + range_4h * 1.1 / 12
    s1 = close_4h_prev - range_4h * 1.1 / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # ===== Volume Spike Filter =====
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # ===== Session Filter (08-20 UTC) =====
    # Pre-compute hour array for session filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close crosses above R1 + above 4h EMA34 + volume spike
            if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and
                close[i] > ema34_4h_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: Close crosses below S1 + below 4h EMA34 + volume spike
            elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and
                  close[i] < ema34_4h_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Close crosses below S1 OR below 4h EMA34
            if close[i] < s1_aligned[i] or close[i] < ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Close crosses above R1 OR above 4h EMA34
            if close[i] > r1_aligned[i] or close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals