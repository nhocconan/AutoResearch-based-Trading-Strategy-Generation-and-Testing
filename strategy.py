#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum with 1d volume spike confirmation and 1w trend filter.
# Long: TRIX crosses above zero + 1d volume > 2x 20-day avg + price above 1w EMA200
# Short: TRIX crosses below zero + 1d volume > 2x 20-day avg + price below 1w EMA200
# TRIX (12,26,9) captures momentum shifts; volume spike confirms institutional interest;
# 1w EMA200 filters for primary trend alignment. Works in bull/bear by requiring trend alignment.
# Target: 80-160 total trades over 4 years (20-40/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    avg_volume_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        avg_volume_1d[i] = np.mean(volume_1d[i-20:i])
    volume_spike = volume_1d > (2.0 * avg_volume_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # 1-week data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 210:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = np.full(len(close_1w), np.nan)
    for i in range(200, len(close_1w)):
        ema_200_1w[i] = np.mean(close_1w[i-200:i])  # Simple MA for EMA200 approximation
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # TRIX (12,26,9) on 4h close
    ema1 = np.full(n, np.nan)
    ema2 = np.full(n, np.nan)
    ema3 = np.full(n, np.nan)
    for i in range(12, n):
        if i == 12:
            ema1[i] = np.mean(close[0:12])
        else:
            ema1[i] = (close[i] * 2/(12+1)) + (ema1[i-1] * (1 - 2/(12+1)))
    for i in range(26, n):
        if i == 26:
            ema2[i] = np.mean(ema1[0:26])
        else:
            ema2[i] = (ema1[i] * 2/(26+1)) + (ema2[i-1] * (1 - 2/(26+1)))
    for i in range(9, n):
        if i == 9:
            ema3[i] = np.mean(ema2[0:9])
        else:
            ema3[i] = (ema2[i] * 2/(9+1)) + (ema3[i-1] * (1 - 2/(9+1)))
    trix = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):
        # Skip if any required data is not ready
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(volume_spike_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trix_now = trix[i]
        trix_prev = trix[i-1]
        vol_spike = volume_spike_aligned[i]
        ema200 = ema_200_1w_aligned[i]
        
        if position == 0:
            # Long: TRIX crosses above zero + volume spike + price above weekly EMA200
            if (trix_prev <= 0 and trix_now > 0 and 
                vol_spike and 
                price > ema200):
                position = 1
                signals[i] = position_size
            # Short: TRIX crosses below zero + volume spike + price below weekly EMA200
            elif (trix_prev >= 0 and trix_now < 0 and 
                  vol_spike and 
                  price < ema200):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TRIX crosses below zero or price below weekly EMA200
            if (trix_now < 0 or
                price < ema200):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: TRIX crosses above zero or price above weekly EMA200
            if (trix_now > 0 or
                price > ema200):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1w_TRIX_Volume_Spike"
timeframe = "4h"
leverage = 1.0