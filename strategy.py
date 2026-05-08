#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Uses 4h price channel breakouts aligned with 12h EMA trend for directional bias,
# and volume spike (>1.5x average) for entry timing. Designed to work in both bull and bear
# markets by following the 12h trend while avoiding false breakouts. Target: 20-50 trades/year.

name = "4h_Donchian_12hEMA50_VolumeBreakout"
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
    
    # Calculate 12h volume average for volume confirmation
    vol_12h = df_12h['volume'].values
    vol_avg_20_12h = np.full(len(vol_12h), np.nan)
    if len(vol_12h) >= 20:
        for i in range(20, len(vol_12h)):
            vol_avg_20_12h[i] = np.mean(vol_12h[i-20:i])
    
    # Align 12h indicators to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_20_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate 4h Donchian channels (20-period)
        highest_high_20 = np.max(high[i-19:i+1])
        lowest_low_20 = np.min(low[i-19:i+1])
        
        # Volume breakout: current 4h volume > 1.5x 20-period average of 12h volume
        vol_breakout = volume[i] > 1.5 * vol_avg_20_12h_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout aligned with 12h EMA trend and volume confirmation
            # Long when price breaks above upper Donchian channel in bullish 12h trend
            long_condition = (
                close[i] > highest_high_20 and          # price breaks above Donchian upper
                close[i] > ema50_12h_aligned[i] and     # price above 12h EMA50 (bullish bias)
                vol_breakout                            # volume confirmation
            )
            
            # Short when price breaks below lower Donchian channel in bearish 12h trend
            short_condition = (
                close[i] < lowest_low_20 and            # price breaks below Donchian lower
                close[i] < ema50_12h_aligned[i] and     # price below 12h EMA50 (bearish bias)
                vol_breakout                            # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below 12h EMA50 or breaks below lower Donchian
            if close[i] < ema50_12h_aligned[i] or close[i] < lowest_low_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above 12h EMA50 or breaks above upper Donchian
            if close[i] > ema50_12h_aligned[i] or close[i] > highest_high_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals