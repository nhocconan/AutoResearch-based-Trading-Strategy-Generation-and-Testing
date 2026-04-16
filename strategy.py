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
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF for daily pivot levels) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1w data (HTF for weekly trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Calculate daily pivot points (based on previous day) ===
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    # R2 = P + (H - L), S2 = P - (H - L)
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    # R4 = 3*P - 2*L, S4 = 3*H - 2*P
    r4 = 3 * pivot - 2 * low_1d
    s4 = 3 * high_1d - 2 * pivot
    
    # Align daily pivot levels to 6h timeframe (use previous day's levels)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # === Weekly trend filter: EMA50 on weekly close ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Volume confirmation: 6h volume > 1.5x 20-period average ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20_6h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ratio_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        pivot_val = pivot_6h[i]
        r1_val = r1_6h[i]
        s1_val = s1_6h[i]
        r2_val = r2_6h[i]
        s2_val = s2_6h[i]
        r3_val = r3_6h[i]
        s3_val = s3_6h[i]
        r4_val = r4_6h[i]
        s4_val = s4_6h[i]
        weekly_trend = ema_50_1w_aligned[i]
        vol_ratio = vol_ratio_6h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below S1 OR weekly trend turns bearish
            if price < s1_val or price < weekly_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above R1 OR weekly trend turns bullish
            if price > r1_val or price > weekly_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume, in weekly uptrend
            if price > r1_val and vol_ratio > 1.5 and price > weekly_trend:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Price breaks below S1 with volume, in weekly downtrend
            elif price < s1_val and vol_ratio > 1.5 and price < weekly_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_WeeklyTrend"
timeframe = "6h"
leverage = 1.0