#!/usr/bin/env python3
# 6h_PivotReversal_R3S3_1dTrend
# Hypothesis: Fade at daily Camarilla R3/S3 levels when 12h trend opposes reversal, with volume confirmation.
# In ranging markets (common in 2025 BTC/ETH), price often reverses at strong daily pivot levels.
# Uses 12h EMA50 as trend filter to avoid counter-trend trades in strong moves.
# Volume spike (>1.5x avg) confirms reversal pressure.
# Target: 20-40 trades/year per symbol with low frequency to minimize fee drag.

name = "6h_PivotReversal_R3S3_1dTrend"
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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # Using previous day's values to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels for each day
    R3 = np.full_like(close_1d, np.nan)
    S3 = np.full_like(close_1d, np.nan)
    
    valid = (~np.isnan(high_1d_prev)) & (~np.isnan(low_1d_prev)) & (~np.isnan(close_1d_prev))
    R3[valid] = close_1d_prev[valid] + (high_1d_prev[valid] - low_1d_prev[valid]) * 1.1 / 4
    S3[valid] = close_1d_prev[valid] - (high_1d_prev[valid] - low_1d_prev[valid]) * 1.1 / 4
    
    # Align to 6h timeframe
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema50_12h[i] = (close_12h[i] * 2 + ema50_12h[i-1] * 48) / 50
    
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need 12h EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        trend_up = close[i] > ema50_12h_aligned[i]
        
        if position == 0:
            # Enter long: price near S3 support + 12h trend up + volume confirmation
            if (close[i] <= S3_6h[i] * 1.005) and trend_up and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: price near R3 resistance + 12h trend down + volume confirmation
            elif (close[i] >= R3_6h[i] * 0.995) and (not trend_up) and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches R3 or trend turns down
            if close[i] >= R3_6h[i] * 0.995 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S3 or trend turns up
            if close[i] <= S3_6h[i] * 1.005 or trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals