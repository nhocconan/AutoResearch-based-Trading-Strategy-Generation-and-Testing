#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # ===== 1d Trend Filter (HTF) =====
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # ===== 12h Camarilla Pivot Levels (LTF) =====
    high_12h = get_htf_data(prices, '12h')['high'].values
    low_12h = get_htf_data(prices, '12h')['low'].values
    close_12h = get_htf_data(prices, '12h')['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    camarilla_S3 = np.zeros(len(close_12h))
    camarilla_R3 = np.zeros(len(close_12h))
    for i in range(len(close_12h)):
        if i == 0:
            camarilla_S3[i] = close_12h[i]
            camarilla_R3[i] = close_12h[i]
        else:
            camarilla_S3[i] = close_12h[i-1] - 1.1 * (high_12h[i-1] - low_12h[i-1]) / 6
            camarilla_R3[i] = close_12h[i-1] + 1.1 * (high_12h[i-1] - low_12h[i-1]) / 6
    
    # Align Camarilla levels to 12h timeframe
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d := get_htf_data(prices, '12h'), camarilla_S3)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    
    # ===== Volume Spike Filter =====
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 + 1d EMA34 uptrend + volume spike
            if (close[i] > camarilla_R3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + 1d EMA34 downtrend + volume spike
            elif (close[i] < camarilla_S3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below S3 OR 1d EMA34 turns down
            if close[i] < camarilla_S3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above R3 OR 1d EMA34 turns up
            if close[i] > camarilla_R3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals