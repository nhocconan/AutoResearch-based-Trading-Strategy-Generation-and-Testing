#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_1dVolume
Hypothesis: In both bull and bear markets, price reacts strongly at Camarilla R3/S3 levels when aligned with weekly trend and confirmed by daily volume spikes. Uses 12h chart for entries with 1w trend filter and 1d volume confirmation to reduce false signals. Works in bull markets by buying pullbacks to S3 in uptrends, and in bear markets by selling rallies to R3 in downtrends.
"""

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_1dVolume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend using EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = np.zeros_like(close_1w)
    for i in range(49, len(close_1w)):
        if i == 49:
            ema_50[i] = np.mean(close_1w[:50])
        else:
            ema_50[i] = (close_1w[i] * 0.039216) + (ema_50[i-1] * 0.960784)
    
    trend_1w = np.where(close_1w > ema_50, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Calculate 1d volume average for spike detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.zeros_like(volume_1d)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_spike_1d = volume_1d > 1.5 * vol_avg_20
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate Camarilla levels for each 12h bar using previous day's OHLC
    # We need to map each 12h bar to the prior trading day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels from prior day
    R3 = np.zeros_like(close_1d)
    S3 = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        range_ = high_1d[i-1] - low_1d[i-1]
        close_prev = close_1d[i-1]
        R3[i] = close_prev + range_ * 1.1 / 4
        S3[i] = close_prev - range_ * 1.1 / 4
    
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_1w_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in direction of weekly trend
        if trend_1w_aligned[i] == 1:  # Weekly uptrend - look for longs at S3
            if position == 0 and close[i] <= S3_aligned[i] * 1.001 and vol_spike_1d_aligned[i] > 0.5:
                # Buy near S3 with volume confirmation
                signals[i] = 0.25
                position = 1
            elif position == 1 and close[i] >= R3_aligned[i] * 0.999:
                # Take profit at R3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else 0.0
                
        elif trend_1w_aligned[i] == -1:  # Weekly downtrend - look for shorts at R3
            if position == 0 and close[i] >= R3_aligned[i] * 0.999 and vol_spike_1d_aligned[i] > 0.5:
                # Sell near R3 with volume confirmation
                signals[i] = -0.25
                position = -1
            elif position == -1 and close[i] <= S3_aligned[i] * 1.001:
                # Take profit at S3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 if position == -1 else 0.0
    
    return signals