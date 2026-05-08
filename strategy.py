#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w trend filter and volume confirmation
# Uses weekly EMA200 trend direction for bias, Donchian(20) breakout for entry,
# and volume > 1.5x average for confirmation. Designed to capture strong trends
# in both bull and bear markets while avoiding false breakouts in choppy conditions.
# Target: 15-30 trades/year on 12h timeframe.

name = "12h_Donchian_20_1wEMA200_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA200 trend filter
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
    
    # Get daily data for volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily 20-period volume average
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            highest_high_20[i] = np.max(high[i-20:i])
            lowest_low_20[i] = np.min(low[i-20:i])
    
    # Align weekly indicators to 12h timeframe
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema200_weekly_aligned[i]) or 
            np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or
            np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average of daily volume
        vol_confirm = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout in direction of weekly trend with volume confirmation
            # Long when price breaks above upper Donchian band in bullish weekly trend
            long_condition = (
                close[i] > highest_high_20[i] and   # price breaks above 20-period high
                close[i] > ema200_weekly_aligned[i] and   # price above weekly EMA200 (bullish bias)
                vol_confirm                         # volume confirmation
            )
            
            # Short when price breaks below lower Donchian band in bearish weekly trend
            short_condition = (
                close[i] < lowest_low_20[i] and    # price breaks below 20-period low
                close[i] < ema200_weekly_aligned[i] and   # price below weekly EMA200 (bearish bias)
                vol_confirm                         # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below the midpoint of Donchian channel
            mid_point = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above the midpoint of Donchian channel
            mid_point = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals