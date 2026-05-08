#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian Breakout + Weekly Trend Filter + Volume Spike
# Uses weekly EMA200 for long-term trend bias, Donchian(20) breakout for entry,
# and volume spike (>1.5x 20-period average) for confirmation. Designed to capture
# major trends while avoiding false breakouts in low-volume conditions. Target: 15-30 trades/year.

name = "6h_Donchian_WeeklyTrend_VolumeSpike"
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
    
    # Get weekly data for trend filter
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
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            highest_high_20[i] = np.max(high[i-20:i+1])
            lowest_low_20[i] = np.min(low[i-20:i+1])
    
    # Calculate volume average (20-period) for volume spike filter
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align weekly EMA200 to 6h timeframe
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
        
        # Volume spike: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout in direction of weekly trend with volume spike
            # Long when price breaks above upper Donchian band in uptrend
            long_condition = (
                close[i] > highest_high_20[i] and   # breakout above Donchian high
                close[i] > ema200_weekly_aligned[i] and   # price above weekly EMA200 (bullish bias)
                vol_spike                           # volume spike for confirmation
            )
            
            # Short when price breaks below lower Donchian band in downtrend
            short_condition = (
                close[i] < lowest_low_20[i] and     # breakout below Donchian low
                close[i] < ema200_weekly_aligned[i] and   # price below weekly EMA200 (bearish bias)
                vol_spike                           # volume spike for confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below lower Donchian band or trend reverses
            if close[i] < lowest_low_20[i] or close[i] < ema200_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above upper Donchian band or trend reverses
            if close[i] > highest_high_20[i] or close[i] > ema200_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals