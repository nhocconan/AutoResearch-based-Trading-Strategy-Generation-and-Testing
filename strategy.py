#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mpf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# Uses weekly EMA20 trend direction for bias, daily Donchian breakout for entry,
# and volume spike (>1.5x average) for confirmation. Designed to capture strong trends
# while avoiding false breakouts in sideways markets. Target: 15-25 trades/year.

name = "1d_Donchian20_20wEMA20_VolumeConfirm"
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
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_weekly = df_weekly['close'].values
    ema20_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 20:
        ema20_weekly[19] = np.mean(close_weekly[:20])
        for i in range(20, len(close_weekly)):
            ema20_weekly[i] = (close_weekly[i] * 2 + ema20_weekly[i-1] * 18) / 20
    
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
    
    # Align weekly EMA20 to daily timeframe
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema20_weekly_aligned[i]) or 
            np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout in direction of weekly trend with volume confirmation
            # Long when price breaks above Donchian upper band in uptrend
            long_condition = (
                close[i] > highest_high_20[i] and   # price breaks above Donchian upper band
                close[i] > ema20_weekly_aligned[i] and   # price above weekly EMA20 (bullish bias)
                vol_confirm                           # volume confirmation
            )
            
            # Short when price breaks below Donchian lower band in downtrend
            short_condition = (
                close[i] < lowest_low_20[i] and    # price breaks below Donchian lower band
                close[i] < ema20_weekly_aligned[i] and   # price below weekly EMA20 (bearish bias)
                vol_confirm                        # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian middle or trend reverses
            donchian_middle = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] < donchian_middle or close[i] < ema20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian middle or trend reverses
            donchian_middle = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] > donchian_middle or close[i] > ema20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals