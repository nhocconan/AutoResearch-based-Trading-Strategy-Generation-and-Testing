#!/usr/bin/env python3
# 2025-06-22 | 4h_PivotReversal_Energy_1dTrend
# Hypothesis: Daily pivot point reversals with 1d EMA100 trend filter and volume confirmation.
# Uses standard pivot point calculation (PP, R1, S1) from previous day's OHLC.
# Long when price crosses above S1 in uptrend (price > EMA100), short when crosses below R1 in downtrend (price < EMA100).
# Volume confirmation (>1.5x 20-period average) filters weak breakouts.
# Designed for low trade frequency (15-30/year) with clear trend alignment to work in both bull and bear markets.

name = "4h_PivotReversal_Energy_1dTrend"
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
    
    # Get daily data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    ph = np.concatenate([[high_1d[0]], high_1d[:-1]])  # previous high
    pl = np.concatenate([[low_1d[0]], low_1d[:-1]])   # previous low
    pc = np.concatenate([[close_1d[0]], close_1d[:-1]]) # previous close
    
    # Calculate daily pivot points (standard formula)
    pp = (ph + pl + pc) / 3.0           # Pivot Point
    r1 = 2 * pp - pl                    # Resistance 1
    s1 = 2 * pp - ph                    # Support 1
    
    # Align daily pivot levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate daily EMA100 for trend filter
    ema_100_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 100:
        ema_100_1d[99] = np.mean(close_1d[0:100])
        for i in range(100, len(close_1d)):
            ema_100_1d[i] = (ema_100_1d[i-1] * 99 + close_1d[i]) / 100
    
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Volume confirmation: current volume / 20-period average volume
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
    
    start_idx = max(20, 100)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price crosses above S1 AND uptrend (price > EMA100) AND volume confirmation
            if (close[i] > s1_aligned[i] and 
                close[i] > ema_100_1d_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below R1 AND downtrend (price < EMA100) AND volume confirmation
            elif (close[i] < r1_aligned[i] and 
                  close[i] < ema_100_1d_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below pivot point OR trend reversal (price < EMA100)
            if close[i] < pp_aligned[i] or close[i] < ema_100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above pivot point OR trend reversal (price > EMA100)
            if close[i] > pp_aligned[i] or close[i] > ema_100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals