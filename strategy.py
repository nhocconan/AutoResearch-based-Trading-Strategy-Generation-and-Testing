#!/usr/bin/env python3
name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
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
    
    # === 1W DATA FOR TREND FILTER (EMA34) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === 1W DATA FOR CAMARILLA LEVELS ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    rng_1w = high_1w - low_1w
    R1_1w = close_1w + rng_1w * 1.1 / 12
    S1_1w = close_1w - rng_1w * 1.1 / 12
    R1_1d = align_htf_to_ltf(prices, df_1w, R1_1w)
    S1_1d = align_htf_to_ltf(prices, df_1w, S1_1w)
    
    # === VOLUME SPIKE (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(R1_1d[i]) or
            np.isnan(S1_1d[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: PRICE BREAKS ABOVE R1 + ABOVE 1W EMA34 + VOLUME SPIKE
            if (close[i] > R1_1d[i] and 
                close[i] > ema34_1w_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: PRICE BREAKS BELOW S1 + BELOW 1W EMA34 + VOLUME SPIKE
            elif (close[i] < S1_1d[i] and 
                  close[i] < ema34_1w_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: PRICE BREAKS BELOW S1 (REVERSAL) OR BELOW 1W EMA34
            if close[i] < S1_1d[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: PRICE BREAKS ABOVE R1 (REVERSAL) OR ABOVE 1W EMA34
            if close[i] > R1_1d[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals