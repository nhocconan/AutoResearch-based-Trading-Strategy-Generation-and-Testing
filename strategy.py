#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Weekly Pivot Reversal with 1d Trend Filter and Volume Spike
# Uses weekly pivot points from previous week to identify reversal zones
# Filters by 1d trend (above/below 200 EMA) to avoid counter-trend trades
# Requires volume spike for confirmation
# Designed for 6h timeframe to capture multi-day swings
# Target: 50-150 total trades over 4 years (12-37/year)
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)

name = "6h_WeeklyPivot_Reversal_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # 1d data for trend filter (200 EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Weekly data for pivot points (previous week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot points: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H, R2 = P+(H-L), S2 = P-(H-L)
    # We use previous week's data to avoid look-ahead
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot points from previous week's OHLC
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot points to 6h timeframe (use previous week's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for 200 EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price at or below S1/S2 support, above 1d EMA200 (uptrend), volume spike
            long_cond = volume_spike[i] and close[i] <= s1_aligned[i] and close[i] > ema_200_1d_aligned[i]
            # Short conditions: price at or above R1/R2 resistance, below 1d EMA200 (downtrend), volume spike
            short_cond = volume_spike[i] and close[i] >= r1_aligned[i] and close[i] < ema_200_1d_aligned[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above R1 (take profit at first resistance)
            if close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below S1 (take profit at first support)
            if close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals