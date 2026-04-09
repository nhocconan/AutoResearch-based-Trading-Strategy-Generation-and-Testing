#!/usr/bin/env python3
# 6h_donchian_weekly_pivot_volume_v2
# Hypothesis: 6h Donchian channel breakout with weekly pivot direction and volume confirmation.
# Enters long when price breaks above 6h Donchian(20) upper band with volume spike and weekly pivot > prior weekly pivot (uptrend).
# Enters short when price breaks below 6h Donchian(20) lower band with volume spike and weekly pivot < prior weekly pivot (downtrend).
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 50-150 trades over 4 years.
# Weekly pivot acts as trend filter: only trade breakouts in direction of weekly trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_weekly_pivot_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d HTF data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point = (high + low + close) / 3
    daily_pivot = (high_1d + low_1d + close_1d) / 3
    
    # Align daily pivot to 6h timeframe (completed 1d candle only)
    daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    # Weekly pivot: average of daily pivots over prior 7 days (completed week)
    # We need to calculate weekly pivot from completed weeks only
    weekly_pivot_raw = pd.Series(daily_pivot).rolling(window=7, min_periods=7).mean().values
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_raw)
    
    # Weekly trend: current weekly pivot > prior weekly pivot (uptrend) or < (downtrend)
    # Use 1-period lag to avoid look-ahead (prior completed week)
    weekly_prior_pivot = np.roll(weekly_pivot_aligned, 1)
    weekly_prior_pivot[0] = np.nan  # First value has no prior
    weekly_uptrend = weekly_pivot_aligned > weekly_prior_pivot
    weekly_downtrend = weekly_pivot_aligned < weekly_prior_pivot
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_prior_pivot[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian lower band
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper band
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian upper band with volume spike and weekly uptrend
            if (close[i] > highest_high[i]) and \
               (vol_spike[i]) and \
               (weekly_uptrend[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower band with volume spike and weekly downtrend
            elif (close[i] < lowest_low[i]) and \
                 (vol_spike[i]) and \
                 (weekly_downtrend[i]):
                position = -1
                signals[i] = -0.25
    
    return signals