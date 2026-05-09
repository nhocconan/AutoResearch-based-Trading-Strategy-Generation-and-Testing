#!/usr/bin/env python3
# 4h_4H_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike_Dyn
# Hypothesis: Camarilla R3/S3 breakouts with 12h EMA50 trend filter and dynamic volume spike filter.
# Works in bull/bear: Trend filter avoids counter-trend trades, volume spike confirms institutional interest.
# R3/S3 levels provide institutional-grade support/resistance for breakouts.
# Uses 12h EMA50 for trend and volume ratio (current/20-bar average) for confirmation.
# Optimized to reduce trade frequency and improve risk-adjusted returns.

name = "4h_4H_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike_Dyn"
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
    
    # Calculate Camarilla levels from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    ph = np.concatenate([[high_1d[0]], high_1d[:-1]])  # previous high
    pl = np.concatenate([[low_1d[0]], low_1d[:-1]])   # previous low
    pc = np.concatenate([[close_1d[0]], close_1d[:-1]]) # previous close
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = pc + (ph - pl) * 1.1 / 4
    camarilla_s3 = pc - (ph - pl) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (ema_50_12h[i-1] * 49 + close_12h[i]) / 50
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 AND uptrend (price > EMA50) AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_ratio[i] > 2.0):  # Increased threshold to reduce trades
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 AND downtrend (price < EMA50) AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_ratio[i] > 2.0):  # Increased threshold to reduce trades
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 OR trend reversal (price < EMA50)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 OR trend reversal (price > EMA50)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals