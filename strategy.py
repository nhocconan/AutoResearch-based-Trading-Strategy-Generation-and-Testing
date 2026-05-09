#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (most significant for breakouts)
    R1 = pivot_1d + (range_1d * 1.0) / 4
    S1 = pivot_1d - (range_1d * 1.0) / 4
    R4 = pivot_1d + (range_1d * 1.5) / 2
    S4 = pivot_1d - (range_1d * 1.5) / 2
    
    # Align daily Camarilla levels to 4h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike detection (4h timeframe)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]  # Require strong volume spike
        
        if position == 0:
            # Long: Price breaks above daily R1 with daily uptrend and volume spike
            if close[i] > R1_1d_aligned[i] and close[i] > ema_1d_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with daily downtrend and volume spike
            elif close[i] < S1_1d_aligned[i] and close[i] < ema_1d_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below daily S1 or trend turns down
            if close[i] < S1_1d_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above daily R1 or trend turns up
            if close[i] > R1_1d_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals