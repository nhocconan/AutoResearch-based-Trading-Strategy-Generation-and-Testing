#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
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
    
    # === 1D DATA FOR TREND FILTER (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 1D DATA FOR CAMARILLA LEVELS (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    rng = high_1d - low_1d
    R1 = close_1d + rng * 1.1 / 12
    S1 = close_1d - rng * 1.1 / 12
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    
    # === VOLUME SPIKE (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(R1_4h[i]) or
            np.isnan(S1_4h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: PRICE BREAKS ABOVE R1 + ABOVE 1D EMA50 + VOLUME SPIKE
            if (close[i] > R1_4h[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: PRICE BREAKS BELOW S1 + BELOW 1D EMA50 + VOLUME SPIKE
            elif (close[i] < S1_4h[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: PRICE BREAKS BELOW S1 (REVERSAL) OR BELOW 1D EMA50
            if close[i] < S1_4h[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: PRICE BREAKS ABOVE R1 (REVERSAL) OR ABOVE 1D EMA50
            if close[i] > R1_4h[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals