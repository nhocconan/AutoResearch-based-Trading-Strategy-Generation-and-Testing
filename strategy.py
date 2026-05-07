#!/usr/bin/env python3
name = "1d_Weekly_Pivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Load daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for weekly pivot calculation
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    # Weekly pivot points (using previous week's data)
    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    r1 = pivot + (prev_weekly_high - prev_weekly_low) * 1.1 / 4
    s1 = pivot - (prev_weekly_high - prev_weekly_low) * 1.1 / 4
    r2 = pivot + (prev_weekly_high - prev_weekly_low) * 1.1 / 2
    s2 = pivot - (prev_weekly_high - prev_weekly_low) * 1.1 / 2
    
    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume spike: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Wait for volume MA and weekly EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike in weekly uptrend
            if close[i] > r1_aligned[i] and vol_spike[i] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume spike in weekly downtrend
            elif close[i] < s1_aligned[i] and vol_spike[i] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below S1 or weekly trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above R1 or weekly trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot point breakouts on daily timeframe with weekly EMA34 trend filter and volume confirmation.
# Long when price breaks above weekly R1 (bullish breakout) with volume spike in weekly uptrend.
# Short when price breaks below weekly S1 (bearish breakdown) with volume spike in weekly downtrend.
# Uses daily timeframe to target 30-100 trades over 4 years (7-25/year), avoiding overtrading.
# Weekly pivot levels provide institutional reference points; breakouts with volume indicate conviction.
# Weekly EMA34 filter ensures trades align with higher-timeframe trend.
# Works in both bull and bear markets by capturing breakouts in the direction of weekly trend.