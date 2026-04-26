#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA_Trend_VolumeSpike_v1
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts with 12h EMA50 trend filter and volume spike capture continuation moves in both bull and bear markets. R1/S1 levels act as intraday support/resistance where breaks indicate short-term momentum. Volume spike confirms breakout validity. 12h EMA50 ensures alignment with medium-term trend. Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter (EMA)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Load 1d data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla R1 and S1 levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 levels
    rang = high_1d - low_1d
    r1 = close_1d + (rang * 1.1 / 12)
    s1 = close_1d - (rang * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike detection on 4h (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 12h trend filter (EMA50)
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Long logic: price breaks above R1 with volume spike + in uptrend
        if close[i] > r1_aligned[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below S1 with volume spike + in downtrend
        elif close[i] < s1_aligned[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite Camarilla level or trend weakens
        elif position == 1 and (close[i] < s1_aligned[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > r1_aligned[i] or not downtrend):
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

name = "4h_Camarilla_R1_S1_Breakout_12hEMA_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0