#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation
# Uses 12h EMA50 for trend direction, Donchian breakout for entry timing, and volume >1.5x average for confirmation.
# Designed to capture strong trending moves while avoiding false breakouts in ranging markets.
# Works in both bull and bear markets by following the 12h trend. Target: 25-40 trades/year.

name = "4h_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema50_12h[i] = (close_12h[i] * 2 + ema50_12h[i-1] * 48) / 50
    
    # Calculate 12h volume average for confirmation
    vol_12h = df_12h['volume'].values
    vol_avg_20_12h = np.full(len(vol_12h), np.nan)
    if len(vol_12h) >= 20:
        for i in range(20, len(vol_12h)):
            vol_avg_20_12h[i] = np.mean(vol_12h[i-20:i])
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            highest_high_20[i] = np.max(high[i-19:i+1])
            lowest_low_20[i] = np.min(low[i-19:i+1])
    
    # Align 12h indicators to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_20_12h_aligned[i]) or
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average of 12h volume
        vol_confirm = volume[i] > 1.5 * vol_avg_20_12h_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout in direction of 12h EMA trend with volume confirmation
            # Long when price breaks above Donchian upper band in uptrend
            long_condition = (
                close[i] > highest_high_20[i] and   # price breaks above Donchian upper band
                close[i] > ema50_12h_aligned[i] and   # price above 12h EMA50 (uptrend)
                vol_confirm                           # volume confirmation
            )
            
            # Short when price breaks below Donchian lower band in downtrend
            short_condition = (
                close[i] < lowest_low_20[i] and    # price breaks below Donchian lower band
                close[i] < ema50_12h_aligned[i] and   # price below 12h EMA50 (downtrend)
                vol_confirm                           # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian middle or trend changes
            mid_band = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] < mid_band or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian middle or trend changes
            mid_band = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] > mid_band or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals