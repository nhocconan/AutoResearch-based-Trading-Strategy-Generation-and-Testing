#/usr/bin/env python3
name = "6h_PriceAction_PivotBreakout_1wTrend_VolumeSpike"
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
    
    # === 1d Data for Pivot Points ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Weekly Pivot Points from previous week ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Standard pivot point calculation
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivots to 6h timeframe
    pivot_1w_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_6h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_6h = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_6h = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_6h = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_6h = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_6h = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # === 1d Trend Filter (EMA 34) ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike Detection (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_6h[i]) or 
            np.isnan(r1_1w_6h[i]) or
            np.isnan(s1_1w_6h[i]) or
            np.isnan(r2_1w_6h[i]) or
            np.isnan(s2_1w_6h[i]) or
            np.isnan(r3_1w_6h[i]) or
            np.isnan(s3_1w_6h[i]) or
            np.isnan(ema34_1d_6h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 + above weekly EMA + volume spike
            if (close[i] > r1_1w_6h[i] and 
                close[i] > ema34_1d_6h[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + below weekly EMA + volume spike
            elif (close[i] < s1_1w_6h[i] and 
                  close[i] < ema34_1d_6h[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S1 or below weekly EMA
            if close[i] < s1_1w_6h[i] or close[i] < ema34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R1 or above weekly EMA
            if close[i] > r1_1w_6h[i] or close[i] > ema34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals