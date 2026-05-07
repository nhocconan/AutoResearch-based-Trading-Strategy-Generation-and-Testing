#!/usr/bin/env python3
name = "6h_WeeklyPivot_Donchian_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load weekly data ONCE before loop for Pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate weekly Pivot (standard) from previous week
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Weekly Pivot support/resistance levels
    s1 = pivot - range_hl
    r1 = pivot + range_hl
    
    # Calculate Donchian channel (20-period) from 4h data
    donchian_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly levels and Donchian to 6h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4, 20)  # Wait for EMA, volume MA, and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above both S1 and Donchian high with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if (close[i] > s1_aligned[i] and close[i] > donchian_high_aligned[i] and 
                vol_condition and uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below both R1 and Donchian low with volume and daily downtrend
            elif (close[i] < r1_aligned[i] and close[i] < donchian_low_aligned[i] and 
                  vol_condition and not uptrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below S1 or Donchian high or volume drops
            if (close[i] < s1_aligned[i] or close[i] < donchian_high_aligned[i] or 
                volume[i] < vol_ma_4[i] * 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above R1 or Donchian low or volume drops
            if (close[i] > r1_aligned[i] or close[i] > donchian_low_aligned[i] or 
                volume[i] < vol_ma_4[i] * 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Weekly Pivot + Donchian breakout with 1d trend and volume confirmation
# - Weekly Pivot S1/R1 act as key support/resistance levels from prior week
# - Donchian(20) breakout provides trend confirmation and structure
# - Long when price breaks above BOTH S1 and Donchian high with volume in daily uptrend
# - Short when price breaks below BOTH R1 and Donchian low with volume in daily downtrend
# - Volume spike (2.0x average) confirms institutional participation
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or Donchian levels or volume weakens
# - Position size 0.25 targets ~30-80 trades/year, avoiding fee drag
# - Novel combination: Weekly Pivot (1w) + Donchian (4h) + trend (1d) + volume (6h)
# - Dual confirmation (Pivot + Donchian) reduces false breakouts significantly
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Aims for 60-160 total trades over 4 years (15-40/year) to stay within limits