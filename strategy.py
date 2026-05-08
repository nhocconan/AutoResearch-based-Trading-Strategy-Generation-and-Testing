#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + Volume confirmation
# Uses weekly EMA50 for trend direction, Donchian channel breakouts for entry,
# and volume > 1.5x average for confirmation. Designed to work in both bull and bear
# markets by following the weekly trend while avoiding false breakouts. Target: 10-25 trades/year.

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 50:
        ema50_weekly[49] = np.mean(close_weekly[:50])
        for i in range(50, len(close_weekly)):
            ema50_weekly[i] = (close_weekly[i] * 2 + ema50_weekly[i-1] * 48) / 50
    
    # Calculate daily Donchian channel (20-period)
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            highest_high_20[i] = np.max(high[i-20:i])
            lowest_low_20[i] = np.min(low[i-20:i])
    
    # Calculate daily volume average for volume confirmation
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align weekly EMA50 to daily timeframe
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or 
            np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for entry: Donchian breakout in direction of weekly trend with volume confirmation
            # Long when price breaks above upper Donchian band in uptrend
            long_condition = (
                close[i] > highest_high_20[i] and   # price breaks above Donchian high
                close[i] > ema50_weekly_aligned[i] and   # price above weekly EMA50 (bullish bias)
                volume[i] > 1.5 * vol_avg_20[i]   # volume confirmation
            )
            
            # Short when price breaks below lower Donchian band in downtrend
            short_condition = (
                close[i] < lowest_low_20[i] and   # price breaks below Donchian low
                close[i] < ema50_weekly_aligned[i] and   # price below weekly EMA50 (bearish bias)
                volume[i] > 1.5 * vol_avg_20[i]   # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian midpoint or trend changes
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] < midpoint or close[i] < ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian midpoint or trend changes
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] > midpoint or close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals