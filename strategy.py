#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout + weekly EMA trend + volume confirmation
# Uses weekly EMA200 for long-term trend filter, 1d Donchian(20) breakouts for entries,
# and volume spike (>1.5x 20-day average) for confirmation. Designed to capture
# strong trending moves while avoiding chop. Target: 10-25 trades/year.

name = "1d_Donchian20_WeeklyEMA200_VolumeConfirm"
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
    if len(df_weekly) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_weekly = df_weekly['close'].values
    ema200_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 200:
        ema200_weekly[199] = np.mean(close_weekly[:200])
        for i in range(200, len(close_weekly)):
            ema200_weekly[i] = (close_weekly[i] * 2 + ema200_weekly[i-1] * 198) / 200
    
    # Calculate daily Donchian channels (20-period)
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
    
    # Align weekly EMA200 to daily timeframe
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema200_weekly_aligned[i]) or np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout in direction of weekly trend with volume confirmation
            # Long when price breaks above Donchian high in uptrend
            long_condition = (
                close[i] > highest_high_20[i] and   # price breaks above Donchian high
                close[i] > ema200_weekly_aligned[i] and   # price above weekly EMA200 (uptrend)
                vol_confirm                         # volume confirmation
            )
            
            # Short when price breaks below Donchian low in downtrend
            short_condition = (
                close[i] < lowest_low_20[i] and    # price breaks below Donchian low
                close[i] < ema200_weekly_aligned[i] and   # price below weekly EMA200 (downtrend)
                vol_confirm                         # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian low or trend reverses
            if close[i] < lowest_low_20[i] or close[i] < ema200_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high or trend reverses
            if close[i] > highest_high_20[i] or close[i] > ema200_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals