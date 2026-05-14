#!/usr/bin/env python3
name = "4h_Camarilla_R3_S4_Breakout_12hEMA34_VolumeSpike"
timeframe = "4h"
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
    
    # === 12H DATA FOR EMA34 TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # === CALCULATE EMA34 FOR TREND FILTER ===
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === CALCULATE CAMARILLA LEVELS (R3, S4) FROM PREVIOUS DAY ===
    # Need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_12h = df_1d['low'].values
    
    rng = high_1d - low_12h
    R3 = close_1d + rng * 1.1 / 4
    S4 = close_1d - rng * 1.1 / 4
    
    # ALIGN TO 4H TIMEFRAME (PREVIOUS DAY'S LEVELS AVAILABLE AT OPEN)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    
    # === VOLUME SPIKE DETECTION (20-PERIOD) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(R3_4h[i]) or
            np.isnan(S4_4h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: PRICE BREAKS ABOVE R3 + ABOVE 12H EMA34 + VOLUME SPIKE
            if (close[i] > R3_4h[i] and 
                close[i] > ema34_12h_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: PRICE BREAKS BELOW S4 + BELOW 12H EMA34 + VOLUME SPIKE
            elif (close[i] < S4_4h[i] and 
                  close[i] < ema34_12h_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: PRICE BREAKS BELOW S4 (REVERSAL) OR BELOW 12H EMA34
            if close[i] < S4_4h[i] or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: PRICE BREAKS ABOVE R3 (REVERSAL) OR ABOVE 12H EMA34
            if close[i] > R3_4h[i] or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals