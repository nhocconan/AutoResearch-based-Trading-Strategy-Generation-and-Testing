#!/usr/bin/env python3
"""
1d_WeeklyPivot_Donchian_Breakout_TrendFilter
Hypothesis: Buy when price breaks above weekly pivot resistance with bullish 1w EMA200 trend and volume confirmation; sell when price breaks below weekly pivot support with bearish 1w EMA200 trend and volume confirmation. Designed for 1d timeframe to capture multi-day trends with low trade frequency, suitable for both bull and bear markets via trend filter.
"""

name = "1d_WeeklyPivot_Donchian_Breakout_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot points (based on prior week)
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    valid_idx = ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close)
    weekly_pivot = np.full_like(prev_close, np.nan)
    weekly_r1 = np.full_like(prev_close, np.nan)
    weekly_s1 = np.full_like(prev_close, np.nan)
    
    weekly_pivot[valid_idx] = (prev_high[valid_idx] + prev_low[valid_idx] + prev_close[valid_idx]) / 3
    weekly_r1[valid_idx] = 2 * weekly_pivot[valid_idx] - prev_low[valid_idx]
    weekly_s1[valid_idx] = 2 * weekly_pivot[valid_idx] - prev_high[valid_idx]
    
    # Align weekly levels to daily timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Weekly EMA200 for trend filter
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price breaks above weekly R1 with bullish trend and volume
            if (weekly_r1_aligned[i] > 0 and not np.isnan(weekly_r1_aligned[i]) and
                high[i] > weekly_r1_aligned[i] and
                close[i] > ema_200_1w_aligned[i] and
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S1 with bearish trend and volume
            elif (weekly_s1_aligned[i] > 0 and not np.isnan(weekly_s1_aligned[i]) and
                  low[i] < weekly_s1_aligned[i] and
                  close[i] < ema_200_1w_aligned[i] and
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly pivot or trend turns bearish
            if (weekly_pivot_aligned[i] > 0 and not np.isnan(weekly_pivot_aligned[i]) and
                (low[i] < weekly_pivot_aligned[i] or close[i] < ema_200_1w_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly pivot or trend turns bullish
            if (weekly_pivot_aligned[i] > 0 and not np.isnan(weekly_pivot_aligned[i]) and
                (high[i] > weekly_pivot_aligned[i] or close[i] > ema_200_1w_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals