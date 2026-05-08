#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_PriceAction_PivotReversal_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for pivot points and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard)
    n1d = len(close_1d)
    pivot = np.full(n1d, np.nan)
    r1 = np.full(n1d, np.nan)
    s1 = np.full(n1d, np.nan)
    r2 = np.full(n1d, np.nan)
    s2 = np.full(n1d, np.nan)
    
    for i in range(1, n1d):
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        pivot[i] = (H + L + C) / 3.0
        r1[i] = 2 * pivot[i] - L
        s1[i] = 2 * pivot[i] - H
        r2[i] = pivot[i] + (high_1d[i-1] - low_1d[i-1])
        s2[i] = pivot[i] - (high_1d[i-1] - low_1d[i-1])
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price bounces from S1/S2 with 1d uptrend + volume
            long_cond = ((close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1]) or
                        (close[i] > s2_aligned[i] and close[i-1] <= s2_aligned[i-1])) and \
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and \
                        volume_filter[i]
            
            # Short: price rejects from R1/R2 with 1d downtrend + volume
            short_cond = ((close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1]) or
                         (close[i] < r2_aligned[i] and close[i-1] >= r2_aligned[i-1])) and \
                        ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and \
                        volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below pivot or shows rejection at R1
            if close[i] < pivot_aligned[i] or (close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above pivot or shows bounce from S1
            if close[i] > pivot_aligned[i] or (close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals