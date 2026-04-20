#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian Channel Breakout with 1d Weekly Pivot Direction Filter
# - Long when price breaks above 6h Donchian upper (20) and price is above 1d weekly pivot (bullish bias)
# - Short when price breaks below 6h Donchian lower (20) and price is below 1d weekly pivot (bearish bias)
# - Volume confirmation: current volume > 1.5x 20-period average volume on 6h
# - Designed to capture breakouts with institutional bias from weekly pivot levels
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week (using last 5 trading days approx)
    # Using prior week's high, low, close to calculate pivot for current week
    # For simplicity, we use 5-day lookback (1 trading week)
    if len(high_1d) >= 5:
        # Get prior week's OHLC (5 days ago to yesterday)
        prior_week_high = np.max(high_1d[:-2]) if len(high_1d) > 2 else high_1d[-1]  # exclude last 2 days (current week forming)
        prior_week_low = np.min(low_1d[:-2]) if len(low_1d) > 2 else low_1d[-1]
        prior_week_close = close_1d[-2] if len(close_1d) > 1 else close_1d[-1]
        
        # Calculate pivot and support/resistance levels
        pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
        r1 = 2 * pivot - prior_week_low
        s1 = 2 * pivot - prior_week_high
        r2 = pivot + (prior_week_high - prior_week_low)
        s2 = pivot - (prior_week_high - prior_week_low)
        r3 = prior_week_high + 2 * (prior_week_low - prior_week_close)
        s3 = prior_week_low - 2 * (prior_week_high - prior_week_close)
    else:
        # Not enough data, use current values
        pivot = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3.0
        r1 = pivot
        s1 = pivot
        r2 = pivot
        s2 = pivot
        r3 = pivot
        s3 = pivot
    
    # We'll use the pivot as bias indicator: price > pivot = bullish bias, price < pivot = bearish bias
    weekly_pivot_bias = pivot
    
    # Align weekly pivot bias to 6h timeframe
    weekly_pivot_bias_array = np.full_like(close_1d, weekly_pivot_bias)
    weekly_pivot_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_bias_array)
    
    # Calculate Donchian Channel (20) on 6h timeframe
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(weekly_pivot_bias_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        volume = volume_6h[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        pivot_bias = weekly_pivot_bias_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper + price above weekly pivot + volume confirmation
            if price > upper and price > pivot_bias and volume > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + price below weekly pivot + volume confirmation
            elif price < lower and price < pivot_bias and volume > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower or volume drops significantly
            if price < lower or volume < 0.5 * volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper or volume drops significantly
            if price > upper or volume < 0.5 * volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_DirectionFilter"
timeframe = "6h"
leverage = 1.0