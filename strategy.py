#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1w Data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Calculate Camarilla R3, S3 levels from previous day ===
    # R3 = Close + 1.1 * (High - Low) * 1.1 / 2
    # S3 = Close - 1.1 * (High - Low) * 1.1 / 2
    range_1d = high_1d - low_1d
    camarilla_R3 = close_1d + 1.1 * range_1d * 1.1 / 2
    camarilla_S3 = close_1d - 1.1 * range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_R3_shifted = np.roll(camarilla_R3, 1)
    camarilla_S3_shifted = np.roll(camarilla_S3, 1)
    camarilla_R3_shifted[0] = np.nan  # First day has no previous day
    camarilla_S3_shifted[0] = np.nan
    
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3_shifted)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3_shifted)
    
    # === 1w EMA34 for trend filter ===
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Volume spike detection (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 34, 20)  # Camarilla needs previous day, EMA34 needs 34, vol needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 + volume spike + 1w uptrend
            if (close[i] > camarilla_R3_aligned[i] and 
                volume_spike[i] and
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + volume spike + 1w downtrend
            elif (close[i] < camarilla_S3_aligned[i] and 
                  volume_spike[i] and
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 or trend changes
            if close[i] < camarilla_S3_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R3 or trend changes
            if close[i] > camarilla_R3_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals