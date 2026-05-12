#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
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
    
    # === 12h Camarilla R3 and S3 levels ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla R3, S3 levels
    rango_12h = high_12h - low_12h
    camarilla_r3_12h = close_12h + (rango_12h * 1.1 / 4)
    camarilla_s3_12h = close_12h - (rango_12h * 1.1 / 4)
    
    camarilla_r3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    
    # === 12h EMA50 trend filter ===
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === 12h Volume spike filter ===
    vol_12h = df_12h['volume'].values
    vol_avg_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = vol_12h > (2.0 * vol_avg_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_12h_aligned[i]) or 
            np.isnan(camarilla_s3_12h_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_spike_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above R3 + above 12h EMA50 + volume spike
            if (close[i] > camarilla_r3_12h_aligned[i] and
                close[i] > ema50_12h_aligned[i] and
                vol_spike_12h_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Close below S3 + below 12h EMA50 + volume spike
            elif (close[i] < camarilla_s3_12h_aligned[i] and
                  close[i] < ema50_12h_aligned[i] and
                  vol_spike_12h_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below S3 or below EMA50
            if close[i] < camarilla_s3_12h_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above R3 or above EMA50
            if close[i] > camarilla_r3_12h_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals