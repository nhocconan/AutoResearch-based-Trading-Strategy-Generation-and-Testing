#!/usr/bin/env python3
name = "12h_Camarilla_R3_S4_Breakout_1dEMA34_VolumeSpike"
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
    
    # === 1d Data for Camarilla pivot and trend ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Calculate Camarilla levels (R3, S4) from previous day ===
    # R3 = close + 1.1 * (high - low) / 2
    # S4 = close - 2 * (high - low)
    rng = high_1d - low_1d
    R3 = close_1d + 1.1 * rng / 2
    S4 = close_1d - 2 * rng
    
    # Align to 12h timeframe (previous day's levels available at open)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S4_12h = align_htf_to_ltf(prices, df_1d, S4)
    
    # === 1d EMA34 for trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume spike detection (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_12h[i]) or 
            np.isnan(S4_12h[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 + above daily EMA34 + volume spike
            if (close[i] > R3_12h[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S4 + below daily EMA34 + volume spike
            elif (close[i] < S4_12h[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S4 (reversal) or below daily EMA34
            if close[i] < S4_12h[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R3 (reversal) or above daily EMA34
            if close[i] > R3_12h[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals