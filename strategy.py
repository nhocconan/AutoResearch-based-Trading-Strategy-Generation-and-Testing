#!/usr/bin/env python3
name = "6h_1w_1d_PivotBreakout_TrendFilter"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot point and levels from previous week
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_range = prev_week_high - prev_week_low
    
    # Weekly S1 and R1
    weekly_s1 = weekly_pivot - weekly_range * 0.382
    weekly_r1 = weekly_pivot + weekly_range * 0.382
    
    # Align weekly levels to 6h timeframe
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly S1 with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            weekly_uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > weekly_s1_aligned[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly R1 with volume and weekly downtrend
            elif close[i] < weekly_r1_aligned[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below weekly S1 or volume drops
            if close[i] < weekly_s1_aligned[i] or volume[i] < vol_ma_4[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above weekly R1 or volume drops
            if close[i] > weekly_r1_aligned[i] or volume[i] < vol_ma_4[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h weekly pivot breakout with daily trend and volume confirmation
# - Weekly S1/R1 act as strong support/resistance levels from prior week
# - Breakout above weekly S1 with volume in weekly uptrend = long opportunity
# - Breakdown below weekly R1 with volume in weekly downtrend = short opportunity
# - Volume spike (1.5x average) confirms participation
# - Uses daily EMA(34) as trend filter to align with longer-term bias
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to weekly S1/R1 or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding excessive fee drag
# - Weekly pivot provides more stable levels than daily for 6h timeframe
# - Designed to avoid overtrading while capturing significant weekly structure breaks