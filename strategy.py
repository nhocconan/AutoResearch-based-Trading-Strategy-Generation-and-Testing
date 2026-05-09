#!/usr/bin/env python3
# 4h_WeeklyPivot_Breakout_1dTrend_Volume
# Hypothesis: Breakout above/below weekly pivot levels (R2/S2) with volume >1.8x 30-bar average and trend filter from 1d EMA50.
# Uses weekly pivot levels as strong support/resistance. In uptrend (price > EMA50), buy breakout above R2; in downtrend (price < EMA50), sell breakdown below S2.
# Volume filter ensures only high-conviction moves trigger entries. Designed for 20-50 trades/year on 4h timeframe.

name = "4h_WeeklyPivot_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50) with proper initialization
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L, S2 = P-(H-L), R2 = P+(H-L)
    pivot = (high_1w + low_1w + close_1w) / 3
    weekly_range = high_1w - low_1w
    weekly_R2 = pivot + weekly_range  # R2 = P + (H-L)
    weekly_S2 = pivot - weekly_range  # S2 = P - (H-L)
    
    # Align weekly pivot levels to 4h timeframe
    weekly_R2_aligned = align_htf_to_ltf(prices, df_1w, weekly_R2)
    weekly_S2_aligned = align_htf_to_ltf(prices, df_1w, weekly_S2)
    
    # Volume filter: 4h volume / 30-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 30:
        vol_ma[29] = np.mean(volume[0:30])
        for i in range(30, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 29 + volume[i]) / 30
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(weekly_R2_aligned[i]) or \
           np.isnan(weekly_S2_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above weekly R2 AND volume confirmation AND bullish trend (price > EMA50)
            if close[i] > weekly_R2_aligned[i] and volume_ratio[i] > 1.8 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below weekly S2 AND volume confirmation AND bearish trend (price < EMA50)
            elif close[i] < weekly_S2_aligned[i] and volume_ratio[i] > 1.8 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below weekly S2 (reversal signal) or trend turns bearish
            if close[i] < weekly_S2_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above weekly R2 (reversal signal) or trend turns bullish
            if close[i] > weekly_R2_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals