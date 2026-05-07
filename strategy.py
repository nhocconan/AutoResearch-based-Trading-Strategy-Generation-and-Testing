#!/usr/bin/env python3
name = "1d_1w_Camarilla_R1S1_Breakout_Trend_Volume_v1"
timeframe = "1d"
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly high/low for trend filter
    weekly_high = df_1w['high'].rolling(window=20, min_periods=20).max().values
    weekly_low = df_1w['low'].rolling(window=20, min_periods=20).min().values
    
    # Align weekly trend to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Daily price action
    daily_high = pd.Series(high).rolling(window=2, min_periods=2).max().values  # yesterday's high
    daily_low = pd.Series(low).rolling(window=2, min_periods=2).min().values    # yesterday's low
    daily_close = pd.Series(close).shift(1).values  # yesterday's close
    
    # Calculate daily Camarilla pivot levels from previous day
    pivot = (daily_high + daily_low + daily_close) / 3
    range_hl = daily_high - daily_low
    
    # Camarilla levels
    s1 = daily_close - (range_hl * 1.08 / 2)
    r1 = daily_close + (range_hl * 1.08 / 2)
    
    # Volume spike detection: 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Wait for weekly trend and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(s1[i]) or np.isnan(r1[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above yesterday's S1 with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            weekly_uptrend = weekly_high_aligned[i] > weekly_high_aligned[i-1]
            
            if close[i] > s1[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below yesterday's R1 with volume and weekly downtrend
            elif close[i] < r1[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Camarilla S1/R1 breakout with weekly trend and volume confirmation
# - Daily Camarilla S1/R1 act as strong support/resistance levels from prior day's range
# - Breakout above S1 with volume in weekly uptrend = long opportunity
# - Breakdown below R1 with volume in weekly downtrend = short opportunity
# - Volume spike (2x average) confirms institutional participation
# - Weekly trend filter ensures we trade with the higher timeframe momentum
# - Works in both bull (buy S1 breaks in weekly uptrend) and bear (sell R1 breaks in weekly downtrend)
# - Exit when price returns to S1/R1 or volume weakens significantly
# - Position size 0.25 targets ~15-25 trades/year, avoiding excessive fee drag
# - Uses actual daily price action for Camarilla calculation (yesterday's H/L/C)
# - Weekly trend uses 20-period high/low to identify directional bias
# - Designed to work in BOTH bull and bear markets via weekly trend filter
# - Conservative entry criteria to limit trades and combat fee drag on 1d timeframe