#!/usr/bin/env python3
name = "12h_Camarilla_R3_S4_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W DATA FOR EMA50 TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === CALCULATE EMA50 FOR TREND FILTER ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 1D DATA FOR CAMARILLA LEVELS (R3, S4) FROM PREVIOUS DAY ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === CALCULATE CAMARILLA LEVELS (R3, S4) FROM PREVIOUS DAY ===
    rng = high_1d - low_1d
    R3 = close_1d + rng * 1.1 / 4
    S4 = close_1d - rng * 1.1 / 4
    
    # ALIGN TO 12H TIMEFRAME (PREVIOUS DAY'S LEVELS AVAILABLE AT OPEN)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S4_12h = align_htf_to_ltf(prices, df_1d, S4)
    
    # === VOLUME SPIKE DETECTION (20-PERIOD) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(R3_12h[i]) or
            np.isnan(S4_12h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: PRICE BREAKS ABOVE R3 + ABOVE 1W EMA50 + VOLUME SPIKE
            if (close[i] > R3_12h[i] and 
                close[i] > ema50_1w_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: PRICE BREAKS BELOW S4 + BELOW 1W EMA50 + VOLUME SPIKE
            elif (close[i] < S4_12h[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: PRICE BREAKS BELOW S4 (REVERSAL) OR BELOW 1W EMA50
            if close[i] < S4_12h[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: PRICE BREAKS ABOVE R3 (REVERSAL) OR ABOVE 1W EMA50
            if close[i] > R3_12h[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals