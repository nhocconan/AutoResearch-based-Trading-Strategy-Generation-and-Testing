#!/usr/bin/env python3
name = "6h_Pivot_Rotation_Squeeze"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 100 or len(df_1w) < 20:
        return np.zeros(n)
    
    # --- Daily Pivot Points (Classic) ---
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate pivots (standard: PP = (H+L+C)/3)
    pivot = (d_high + d_low + d_close) / 3.0
    r1 = 2 * pivot - d_low
    s1 = 2 * pivot - d_high
    r2 = pivot + (d_high - d_low)
    s2 = pivot - (d_high - d_low)
    r3 = d_high + 2 * (pivot - d_low)
    s3 = d_low - 2 * (d_high - pivot)
    
    # --- Daily Bollinger Width for Squeeze Detection ---
    d_close_series = pd.Series(d_close)
    bb_mid = d_close_series.rolling(20, min_periods=20).mean()
    bb_std = d_close_series.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    bb_width_values = bb_width.values
    
    # --- Daily Volume Average (20-period) ---
    d_volume = df_1d['volume'].values
    d_vol_series = pd.Series(d_volume)
    vol_ma20 = d_vol_series.rolling(20, min_periods=20).mean().values
    
    # --- 6-period Volume Spike Detection (for entry timing) ---
    vol_series = pd.Series(volume)
    vol_ma6 = vol_series.rolling(6, min_periods=6).mean().values
    
    # --- Weekly Trend Filter (EMA 34) ---
    w_close = df_1w['close'].values
    w_ema34 = pd.Series(w_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_up = w_close > w_ema34
    
    # --- Align all daily indicators to 6h ---
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width_values)
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    weekly_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 6)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(bb_width_aligned[i]) or np.isnan(vol_ma20_aligned[i]) or np.isnan(weekly_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Squeeze condition: Bollinger Width below 20th percentile (low volatility)
        # Calculate percentile dynamically using lookback
        lookback = min(i, 100)
        if lookback >= 20:
            bb_width_slice = bb_width_aligned[max(0, i-lookback):i+1]
            bb_width_percentile = np.percentile(bb_width_slice, 20)
            squeeze = bb_width_aligned[i] <= bb_width_percentile
        else:
            squeeze = False
        
        # Volume confirmation: current volume > 1.5x daily average volume
        vol_confirm = volume[i] > 1.5 * vol_ma20_aligned[i]
        
        if position == 0:
            # Long setup: squeeze + weekly uptrend + price above R1 + volume spike
            if (squeeze and weekly_up_aligned[i] and 
                close[i] > r1_aligned[i] and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short setup: squeeze + weekly downtrend + price below S1 + volume spike
            elif (squeeze and not weekly_up_aligned[i] and 
                  close[i] < s1_aligned[i] and vol_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches R2 or weekly trend changes or squeeze breaks
            if (close[i] >= r2_aligned[i] or not weekly_up_aligned[i] or not squeeze):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S2 or weekly trend changes or squeeze breaks
            if (close[i] <= s2_aligned[i] or weekly_up_aligned[i] or not squeeze):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals