#!/usr/bin/env python3
name = "6h_1d_Donchian_Breakout_WeeklyPivot_Trend"
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
    
    # Load weekly data for pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot levels from previous week
    prev_high_w = df_1w['high'].shift(1).values
    prev_low_w = df_1w['low'].shift(1).values
    prev_close_w = df_1w['close'].shift(1).values
    
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    range_w = prev_high_w - prev_low_w
    
    # Weekly pivot levels (standard)
    s1_w = prev_close_w - (range_w * 1.08 / 2)
    r1_w = prev_close_w + (range_w * 1.08 / 2)
    s2_w = prev_close_w - (range_w * 1.16 / 2)
    r2_w = prev_close_w + (range_w * 1.16 / 2)
    s3_w = prev_close_w - (range_w * 1.26 / 4)
    r3_w = prev_close_w + (range_w * 1.26 / 4)
    
    # Align weekly pivot to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_1w, s3_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_1w, r3_w)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channel on 6h (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: 6-period average (1 day of 6h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 6)  # Wait for EMA, Donchian, volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_6[i]) or
            np.isnan(s1_w_aligned[i]) or np.isnan(r1_w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high + weekly pivot support + daily uptrend + volume
            vol_condition = volume[i] > vol_ma_6[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            above_pivot = close[i] > s1_w_aligned[i]  # Above weekly S1 support
            
            if close[i] > high_20[i] and vol_condition and uptrend and above_pivot:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + weekly pivot resistance + daily downtrend + volume
            elif close[i] < low_20[i] and vol_condition and not uptrend and close[i] < r1_w_aligned[i]:  # Below weekly R1 resistance
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or volume drops
            if close[i] < low_20[i] or volume[i] < vol_ma_6[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high or volume drops
            if close[i] > high_20[i] or volume[i] < vol_ma_6[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Donchian breakout with weekly pivot filter and daily trend
# - Donchian(20) breakout captures momentum on 6h timeframe
# - Weekly pivot S1/R1 acts as major support/resistance - only trade in direction of pivot
# - Daily EMA(34) ensures alignment with higher timeframe trend
# - Volume spike (2x average) confirms institutional participation
# - Works in bull (buy breaks above Donchian high in uptrend above weekly pivot)
# - Works in bear (sell breaks below Donchian low in downtrend below weekly pivot)
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Weekly pivot provides stronger filter than daily for 6h timeframe
# - Weekly pivot levels are more significant and less noisy than daily
# - Exit when price returns to Donchian channel or volume weakens