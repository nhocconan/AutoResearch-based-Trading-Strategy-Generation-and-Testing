#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above Camarilla R3 (1d) AND price > 1w EMA50 AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below Camarilla S3 (1d) AND price < 1w EMA50 AND 1d volume > 2.0 * 20-period average volume.
# Exit when price crosses Camarilla pivot point (1d) or opposing S1/R1 level.
# Uses discrete position sizing (0.30) to limit fee churn. Designed for BTC/ETH robustness by capturing institutional breakouts with volume confirmation in trending markets.
# Target: 100-150 total trades over 4 years (25-37/year) for 12h timeframe.

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_1dVolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Camarilla pivot levels from 1d (HTF)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 4.0)
    s3 = pivot - (range_1d * 1.1 / 4.0)
    r1 = pivot + (range_1d * 1.1 / 12.0)
    s1 = pivot - (range_1d * 1.1 / 12.0)
    
    # Align Camarilla levels to lower timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND price > 1w EMA50 AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # SHORT: price breaks below S3 AND price < 1w EMA50 AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below pivot or S1
            if close[i] < pivot_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price crosses above pivot or R1
            if close[i] > pivot_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals