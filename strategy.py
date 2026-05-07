#!/usr/bin/env python3
name = "12h_1d_1w_Camarilla_Pivot_Breakout_Trend"
timeframe = "12h"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot levels from previous week
    prev_high_w = df_1w['high'].shift(1).values
    prev_low_w = df_1w['low'].shift(1).values
    prev_close_w = df_1w['close'].shift(1).values
    
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    range_w = prev_high_w - prev_low_w
    
    # Weekly Camarilla levels
    s1_w = prev_close_w - (range_w * 1.08 / 2)
    r1_w = prev_close_w + (range_w * 1.08 / 2)
    
    # Calculate daily pivot levels from previous day
    prev_high_d = df_1d['high'].shift(1).values
    prev_low_d = df_1d['low'].shift(1).values
    prev_close_d = df_1d['close'].shift(1).values
    
    pivot_d = (prev_high_d + prev_low_d + prev_close_d) / 3
    range_d = prev_high_d - prev_low_d
    
    # Daily Camarilla levels
    s1_d = prev_close_d - (range_d * 1.08 / 2)
    r1_d = prev_close_d + (range_d * 1.08 / 2)
    
    # Align weekly levels to 12h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    
    # Align daily levels to 12h timeframe
    pivot_d_aligned = align_htf_to_ltf(prices, df_1d, pivot_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1_d)
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1_d)
    
    # Weekly trend filter: EMA(21) on weekly close
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 4-period average (2 days of 12h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or
            np.isnan(s1_d_aligned[i]) or np.isnan(r1_d_aligned[i]) or
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend alignment: both weekly and daily must agree
        weekly_uptrend = ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1]
        daily_uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
        trend_aligned = weekly_uptrend and daily_uptrend
        trend_aligned_down = (not weekly_uptrend) and (not daily_uptrend)
        
        vol_condition = volume[i] > vol_ma_4[i] * 2.0
        
        if position == 0:
            # Long: price above both weekly and daily S1 with volume and uptrend
            if (close[i] > s1_w_aligned[i] and close[i] > s1_d_aligned[i] and 
                vol_condition and trend_aligned):
                signals[i] = 0.25
                position = 1
            # Short: price below both weekly and daily R1 with volume and downtrend
            elif (close[i] < r1_w_aligned[i] and close[i] < r1_d_aligned[i] and 
                  vol_condition and trend_aligned_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below either S1 or volume drops
            if (close[i] < s1_w_aligned[i] or close[i] < s1_d_aligned[i] or 
                volume[i] < vol_ma_4[i] * 1.3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above either R1 or volume drops
            if (close[i] > r1_w_aligned[i] or close[i] > r1_d_aligned[i] or 
                volume[i] < vol_ma_4[i] * 1.3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla S1/R1 breakout with weekly/daily trend alignment and volume confirmation
# - Weekly and daily Camarilla S1/R1 act as strong support/resistance levels
# - Breakout above BOTH weekly and daily S1 with volume in aligned uptrend = long opportunity
# - Breakdown below BOTH weekly and daily R1 with volume in aligned downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Requires trend alignment on BOTH weekly and daily timeframes to avoid counter-trend trades
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to either S1/R1 level or volume weakens
# - Position size 0.25 targets ~15-30 trades/year, avoiding fee drag
# - Uses higher timeframe confluence (weekly + daily) for stronger signal quality
# - Designed to work in BOTH bull and bear markets via multi-timeframe trend filter