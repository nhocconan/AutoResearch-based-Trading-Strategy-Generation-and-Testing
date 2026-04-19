#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_AggroDefense_Pivot"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for Pivot and 1d EMA ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day Pivot, R1, S1
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1_1d = 2 * pivot_1d - prev_low_1d
    s1_1d = 2 * pivot_1d - prev_high_1d
    
    # === 1w data for weekly trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA(21) for trend
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align all to 6h timeframe
    pivot_1d_a = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_a = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_a = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_21_1w_a = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1d_a[i]) or np.isnan(r1_1d_a[i]) or np.isnan(s1_1d_a[i]) or np.isnan(ema_21_1w_a[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Weekly trend: price above/below weekly EMA21
        weekly_uptrend = price > ema_21_1w_a[i]
        weekly_downtrend = price < ema_21_1w_a[i]
        
        if position == 0:
            # Long: price breaks above R1d with volume AND weekly uptrend
            if price > r1_1d_a[i] and volume_ok and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1d with volume AND weekly downtrend
            elif price < s1_1d_a[i] and volume_ok and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to pivot (mean reversion to mean)
            if price < pivot_1d_a[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to pivot (mean reversion to mean)
            if price > pivot_1d_a[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals