# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1-week and 1-day higher timeframe filters.
- Uses weekly trend filter (price above/below weekly 50 EMA) to align with long-term trend.
- Uses daily Camarilla pivot levels for mean reversion entries within the weekly trend.
- Goes long when price touches S1/S2 in uptrend, short when price touches R1/R2 in downtrend.
- Volume confirmation (current volume > 1.5x 20-period average) to filter false touches.
- Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
- Targets 15-35 trades/year to avoid fee drag.
"""

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
    
    # === Higher Timeframe Data ===
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Weekly Trend Filter (EMA50) ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily Camarilla Pivot Levels ===
    # Calculate pivots from previous day's OHLC
    # H, L, C from previous day
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Camarilla levels
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12
    r2 = pivot + (prev_high - prev_low) * 1.1 / 6
    r3 = pivot + (prev_high - prev_low) * 1.1 / 4
    r4 = pivot + (prev_high - prev_low) * 1.1 / 2
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12
    s2 = pivot - (prev_high - prev_low) * 1.1 / 6
    s3 = pivot - (prev_high - prev_low) * 1.1 / 4
    s4 = pivot - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 6h timeframe (use previous day's pivots for current day)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === Volume Confirmation (6h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        weekly_trend = ema_50_1w_aligned[i]
        vol_ok = vol_ratio[i] > 1.5
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below S1 or weekly trend turns down
            if price < s1_aligned[i] or price < weekly_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above R1 or weekly trend turns up
            if price > r1_aligned[i] or price > weekly_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: In uptrend (price above weekly EMA50), price touches S1 or S2 with volume
            if price > weekly_trend and vol_ok:
                if abs(price - s1_aligned[i]) < (high[i] - low[i]) * 0.1 or \
                   abs(price - s2_aligned[i]) < (high[i] - low[i]) * 0.1:
                    signals[i] = 0.25
                    position = 1
                    continue
            
            # SHORT: In downtrend (price below weekly EMA50), price touches R1 or R2 with volume
            elif price < weekly_trend and vol_ok:
                if abs(price - r1_aligned[i]) < (high[i] - low[i]) * 0.1 or \
                   abs(price - r2_aligned[i]) < (high[i] - low[i]) * 0.1:
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

name = "6h_Camarilla_S1S2_R1R2_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0