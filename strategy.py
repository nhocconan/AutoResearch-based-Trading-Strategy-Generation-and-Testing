#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_Dyn
Hypothesis: Uses Camarilla pivot R1/S1 breakouts with 1d EMA34 trend filter and volume spike confirmation.
Works in both bull and bear markets by combining mean-reversion breakouts with trend alignment.
Target: 20-40 trades/year to minimize fee drag while capturing high-probability moves.
"""

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_Dyn"
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
    
    # === 1D Data for Camarilla Pivots and Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels (R1, S1) from previous day
    # Camarilla: R1 = C + 1.1*(H-L)/12, S1 = C - 1.1*(H-L)/12
    cam_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    cam_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, cam_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, cam_s1)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers EMA34 and volume MA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with uptrend and volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with downtrend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below S1 (mean reversion) or trend failure
            if close[i] < s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: break above R1 (mean reversion) or trend failure
            if close[i] > r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals