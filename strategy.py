#!/usr/bin/env python3
name = "6h_1d_1w_Camarilla_Pivot_Breakout_Trend"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels from previous week
    prev_high_w = df_1w['high'].shift(1).values
    prev_low_w = df_1w['low'].shift(1).values
    prev_close_w = df_1w['close'].shift(1).values
    
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    range_w = prev_high_w - prev_low_w
    
    # Weekly Camarilla levels
    s1_w = prev_close_w - (range_w * 1.08 / 2)
    r1_w = prev_close_w + (range_w * 1.08 / 2)
    s2_w = prev_close_w - (range_w * 1.16 / 2)
    r2_w = prev_close_w + (range_w * 1.16 / 2)
    s3_w = prev_close_w - (range_w * 1.26 / 4)
    r3_w = prev_close_w + (range_w * 1.26 / 4)
    
    # Calculate daily Camarilla pivot levels from previous day
    prev_high_d = df_1d['high'].shift(1).values
    prev_low_d = df_1d['low'].shift(1).values
    prev_close_d = df_1d['close'].shift(1).values
    
    pivot_d = (prev_high_d + prev_low_d + prev_close_d) / 3
    range_d = prev_high_d - prev_low_d
    
    # Daily Camarilla levels
    s1_d = prev_close_d - (range_d * 1.08 / 2)
    r1_d = prev_close_d + (range_d * 1.08 / 2)
    s2_d = prev_close_d - (range_d * 1.16 / 2)
    r2_d = prev_close_d + (range_d * 1.16 / 2)
    s3_d = prev_close_d - (range_d * 1.26 / 4)
    r3_d = prev_close_d + (range_d * 1.26 / 4)
    
    # Align weekly and daily levels to 6h timeframe
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1_d)
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1_d)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_d_aligned[i]) or 
            np.isnan(r1_d_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 daily with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s1_d_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 daily with volume and daily downtrend
            elif close[i] < r1_d_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 daily or volume drops
            if close[i] < s1_d_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 daily or volume drops
            if close[i] > r1_d_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Camarilla S1/R1 breakout with daily trend and volume confirmation
# - Daily Camarilla S1/R1 act as strong support/resistance levels
# - Breakout above S1 with volume in daily uptrend = long opportunity
# - Breakdown below R1 with volume in daily downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses actual daily Camarilla levels for better responsiveness
# - Designed to work in BOTH bull and bear markets via trend filter
# - Weekly pivot used for context but not direct trigger (focus on daily for responsiveness)