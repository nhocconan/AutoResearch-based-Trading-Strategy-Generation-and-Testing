#!/usr/bin/env python3
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
    
    # === 1d data (HTF for pivot levels and trend) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly pivot points from 1d data (based on prior week)
    # Calculate weekly high/low/close from daily data
    # We'll approximate using rolling window of 5 days for weekly pivot
    # In practice, we'd need actual weekly data, but we'll use daily to approximate
    # For true weekly pivot, we should use weekly data, but per rules we can use 1d as HTF
    # Let's use actual weekly data via 1w timeframe
    
    # === 1w data (HTF for weekly pivot) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    
    # Calculate weekly pivot
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Support and resistance levels
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # === 1d data for trend filter (EMA20) ===
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # === Volume spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma_20 + 1e-10)  # Avoid division by zero
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        pivot = pivot_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        r2 = r2_1w_aligned[i]
        s2 = s2_1w_aligned[i]
        r3 = r3_1w_aligned[i]
        s3 = s3_1w_aligned[i]
        ema_20_1d_val = ema_20_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below weekly S1 or RSI-like condition (using price vs EMA)
            if (price < s1) or (price < ema_20_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above weekly R1 or price > EMA
            if (price > r1) or (price > ema_20_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above weekly R1 with volume and above weekly EMA
                if (price > r1) and (price > ema_20_1d_val) and (vol_ratio_val > 1.5):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below weekly S1 with volume and below weekly EMA
                elif (price < s1) and (price < ema_20_1d_val) and (vol_ratio_val > 1.5):
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

name = "6h_WeeklyPivot_R1S1_Breakout_EMA20_Volume"
timeframe = "6h"
leverage = 1.0