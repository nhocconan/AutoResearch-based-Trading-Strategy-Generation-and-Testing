# 6h_WeeklyPivot_RangeBreakout_Volume
# Hypothesis: Weekly pivot levels act as strong support/resistance zones. When price breaks above weekly R1 or below weekly S1 with volume confirmation and 1d trend alignment, it signals continuation of the breakout move. Works in both bull and bear markets as pivots adapt to price levels. Uses 6h timeframe to capture multi-day moves with lower frequency (target: 20-50 trades/year) to minimize fee drag.
# Entry: Long when price breaks above weekly R1 with volume spike and price above 1d EMA50. Short when price breaks below weekly S1 with volume spike and price below 1d EMA50.
# Exit: When price crosses back below/above weekly pivot point (PP) or 1d EMA50.

#!/usr/bin/env python3
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
    
    # Get weekly data for pivot calculation (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # PP = (H + L + C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pp = np.full(len(high_1w), np.nan)
    r1 = np.full(len(high_1w), np.nan)
    s1 = np.full(len(high_1w), np.nan)
    
    for i in range(len(high_1w)):
        pp[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        r1[i] = 2 * pp[i] - low_1w[i]
        s1[i] = 2 * pp[i] - high_1w[i]
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        alpha = 2 / (50 + 1)
        ema_50_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Align weekly pivots and daily EMA to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 6
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period: need at least 1 weekly bar and 6 volume period
    start_idx = max(vol_period, 1) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume and above 1d EMA50
            if price > r1_aligned[i] and vol_filter and price > ema_50_1d_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below weekly S1 with volume and below 1d EMA50
            elif price < s1_aligned[i] and vol_filter and price < ema_50_1d_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below weekly PP or below 1d EMA50
            if price < pp_aligned[i] or price < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above weekly PP or above 1d EMA50
            if price > pp_aligned[i] or price > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_RangeBreakout_Volume"
timeframe = "6h"
leverage = 1.0