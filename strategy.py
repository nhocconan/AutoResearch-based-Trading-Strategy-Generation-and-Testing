#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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
    
    # === 1W DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === 1D DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # H, L, C from previous day
    H_1d = np.roll(high_1d, 1)
    L_1d = np.roll(low_1d, 1)
    C_1d = np.roll(close_1d, 1)
    # First day will have NaN from roll, handled by checks later
    
    pivot = (H_1d + L_1d + C_1d) / 3.0
    range_hl = H_1d - L_1d
    
    # Camarilla levels
    R1 = C_1d + range_hl * 1.1 / 12
    S1 = C_1d - range_hl * 1.1 / 12
    
    # Align to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    
    # === VOLUME SPIKE (24-period for 12h) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Wait for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_12h[i]) or 
            np.isnan(S1_12h[i]) or
            np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close breaks above R1 + above weekly EMA34 + volume spike
            if (close[i] > R1_12h[i] and 
                close[i] > ema34_1w_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 + below weekly EMA34 + volume spike
            elif (close[i] < S1_12h[i] and 
                  close[i] < ema34_1w_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Close breaks below S1 OR below weekly EMA34
            if close[i] < S1_12h[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close breaks above R1 OR above weekly EMA34
            if close[i] > R1_12h[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals