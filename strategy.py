#!/usr/bin/env python3
# 6H_1W_1D_Donchian20_WeeklyTrend_Pullback
# Hypothesis: On 6h timeframe, enter long when price pulls back to the 20-period Donchian middle during a weekly uptrend, with volume confirmation.
# Short when price pulls back to the Donchian middle during a weekly downtrend.
# Weekly trend filter avoids counter-trend trades, reducing whipsaw in bear markets.
# Pullback to Donchian middle provides favorable risk-reward with confluence.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years).

name = "6H_1W_1D_Donchian20_WeeklyTrend_Pullback"
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
    
    # Get weekly data for trend filter (Donchian 20 on weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly Donchian 20: upper = max(high, 20), lower = min(low, 20), middle = (upper + lower)/2
    # For trend, we use close-based Donchian: highest close and lowest close over 20 weeks
    highest_close_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).max().values
    lowest_close_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).min().values
    middle_donchian_20 = (highest_close_20 + lowest_close_20) / 2
    
    # Get daily data for Donchian 20 (high/low based) for pullback levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Daily Donchian 20: based on high/low
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_donchian_20_1d = (highest_high_20 + lowest_low_20) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align weekly trend to 6h
    weekly_trend_up = highest_close_20 > lowest_close_20  # uptrend if higher highs
    weekly_trend_up_series = pd.Series(weekly_trend_up).values
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up_series)
    
    # Align daily Donchian middle to 6h
    daily_middle_aligned = align_htf_to_ltf(prices, df_1d, middle_donchian_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(daily_middle_aligned[i]) or np.isnan(weekly_trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price near daily Donchian middle (+/- 0.5%) + weekly uptrend + volume confirmation
            price_to_middle_ratio = close[i] / daily_middle_aligned[i]
            if 0.995 <= price_to_middle_ratio <= 1.005 and weekly_trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price near daily Donchian middle (+/- 0.5%) + weekly downtrend + volume confirmation
            elif 0.995 <= price_to_middle_ratio <= 1.005 and not weekly_trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks above weekly Donchian 20 upper (trend continuation)
            # For exit, we need weekly Donchian upper aligned
            weekly_upper = highest_close_20
            weekly_upper_aligned = align_htf_to_ltf(prices, df_1w, weekly_upper)
            if not np.isnan(weekly_upper_aligned[i]) and close[i] > weekly_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks below weekly Donchian 20 lower
            weekly_lower = lowest_close_20
            weekly_lower_aligned = align_htf_to_ltf(prices, df_1w, weekly_lower)
            if not np.isnan(weekly_lower_aligned[i]) and close[i] < weekly_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals