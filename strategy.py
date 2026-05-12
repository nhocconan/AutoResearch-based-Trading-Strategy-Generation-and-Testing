#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation"
timeframe = "6h"
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
    
    # === WEEKLY DATA FOR PIVOT DIRECTION AND DONCHIAN ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === DAILY DATA FOR VOLUME CONFIRMATION ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # === CALCULATE WEEKLY PIVOT POINT (CLASSIC) ===
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # === CALCULATE WEEKLY DONCHIAN CHANNEL (20) ===
    donch_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # === CALCULATE DAILY VOLUME SPIKE (20) ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    
    # === ALIGN WEEKLY INDICATORS TO 6H TIMEFRAME ===
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    
    # === ALIGN DAILY VOLUME SPIKE TO 6H TIMEFRAME ===
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian and volume MA need 20 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or
            np.isnan(donch_low_20_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: PRICE BREAKS ABOVE WEEKLY DONCHIAN HIGH + ABOVE WEEKLY PIVOT + DAILY VOLUME SPIKE
            if (close[i] > donch_high_20_aligned[i] and 
                close[i] > pivot_1w_aligned[i] and
                volume_spike_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: PRICE BREAKS BELOW WEEKLY DONCHIAN LOW + BELOW WEEKLY PIVOT + DAILY VOLUME SPIKE
            elif (close[i] < donch_low_20_aligned[i] and 
                  close[i] < pivot_1w_aligned[i] and
                  volume_spike_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: PRICE BREAKS BELOW WEEKLY DONCHIAN LOW OR BELOW WEEKLY PIVOT
            if close[i] < donch_low_20_aligned[i] or close[i] < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: PRICE BREAKS ABOVE WEEKLY DONCHIAN HIGH OR ABOVE WEEKLY PIVOT
            if close[i] > donch_high_20_aligned[i] or close[i] > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals