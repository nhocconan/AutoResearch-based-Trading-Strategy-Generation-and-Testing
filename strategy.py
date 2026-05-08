#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Breakout_TrendFilter"
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
    
    # Weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly high, low, close for pivot points
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Weekly EMA34 for trend filter
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema34 = weekly_close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    # Volume spike: current volume > 2.0x 20-period average (more strict)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(weekly_ema34_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and weekly uptrend
            long_cond = (close[i] > r3_aligned[i] and 
                        weekly_ema34_aligned[i] > weekly_ema34_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below S3 with volume spike and weekly downtrend
            short_cond = (close[i] < s3_aligned[i] and 
                         weekly_ema34_aligned[i] < weekly_ema34_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below pivot or weekly trend turns down
            if close[i] < pivot_aligned[i] or weekly_ema34_aligned[i] < weekly_ema34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above pivot or weekly trend turns up
            if close[i] > pivot_aligned[i] or weekly_ema34_aligned[i] > weekly_ema34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals