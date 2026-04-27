#!/usr/bin/env python3
"""
4h_1D_Camarilla_R1S1_Breakout_VolumeSpike
Hypothesis: Price breaks above/below Camarilla R1/S1 levels (from daily) with volume spike (2x average) and price above/below 1d EMA50 for trend filter.
Camarilla levels provide institutional support/resistance that works in both bull and bear markets.
Volume spike confirms institutional participation. EMA50 filter ensures we trade with the daily trend.
Target: 20-35 trades/year to stay under fee drag threshold while capturing high-probability moves.
"""

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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Formula: R4 = C + (H-L)*1.5/2, R3 = C + (H-L)*1.25/2, R2 = C + (H-L)*1.1/2, R1 = C + (H-L)*1.05/2
    #          S1 = C - (H-L)*1.05/2, S2 = C - (H-L)*1.1/2, S3 = C - (H-L)*1.25/2, S4 = C - (H-L)*1.5/2
    # where C = (H+L+CLOSE)/3 of previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivots = (high_1d + low_1d + close_1d) / 3
    ranges = high_1d - low_1d
    
    r1 = pivots + ranges * 1.05 / 2
    s1 = pivots - ranges * 1.05 / 2
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_1d_period:
        ema_1d[ema_1d_period - 1] = np.mean(close_1d[:ema_1d_period])
        multiplier = 2 / (ema_1d_period + 1)
        for i in range(ema_1d_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 50)  # volume MA needs 20, EMA needs 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Trend filter: price vs 1d EMA50
        uptrend = price > ema_1d_aligned[i]
        downtrend = price < ema_1d_aligned[i]
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long: break above R1 with volume and uptrend
            if uptrend and volume_confirmation and price > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and downtrend
            elif downtrend and volume_confirmation and price < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price below S1 or trend reversal
            if price < s1_aligned[i] or price <= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price above R1 or trend reversal
            if price > r1_aligned[i] or price >= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_1D_Camarilla_R1S1_Breakout_VolumeSpike"
timeframe = "4h"
leverage = 1.0