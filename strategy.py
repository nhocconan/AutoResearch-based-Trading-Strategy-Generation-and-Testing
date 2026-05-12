#!/usr/bin/env python3
name = "4h_Camarilla_R4_S3_Breakout_1dEMA34_VolumeSpike_v2"
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
    
    # === 1d Data for EMA34 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === Calculate EMA34 for trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Calculate Camarilla levels (R4, S3) from previous day ===
    rng = high_1d - low_1d
    R4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align to 4h timeframe (previous day's levels available at open)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    
    # === Volume spike detection (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(R4_4h[i]) or
            np.isnan(S3_4h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R4 + above 1d EMA34 + volume spike
            if (close[i] > R4_4h[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + below 1d EMA34 + volume spike
            elif (close[i] < S3_4h[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 (reversal) or below 1d EMA34
            if close[i] < S3_4h[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R4 (reversal) or above 1d EMA34
            if close[i] > R4_4h[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals