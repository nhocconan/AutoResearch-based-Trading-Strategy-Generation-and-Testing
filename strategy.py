#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA200 trend filter and volume confirmation
# Uses 12h EMA200 for trend direction, Donchian breakout for entry timing, and volume spike (>1.5x) for confirmation.
# Works in bull markets (buy breakouts above EMA200) and bear markets (sell breakdowns below EMA200).
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_Donchian_12hEMA200_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    # Calculate 12h EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema200_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 200:
        ema200_12h[199] = np.mean(close_12h[:200])
        for i in range(200, len(close_12h)):
            ema200_12h[i] = (close_12h[i] * 2 + ema200_12h[i-1] * 198) / 200
    
    # Align 12h EMA200 to 6h timeframe
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
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
        if np.isnan(ema200_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels (20-period) for current 6h
        highest_high = np.max(high[i-19:i+1])
        lowest_low = np.min(low[i-19:i+1])
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average volume
        vol_avg_20 = np.mean(volume[i-20:i]) if i >= 20 else 0
        vol_confirm = volume[i] > 1.5 * vol_avg_20 if vol_avg_20 > 0 else False
        
        if position == 0:
            # Look for entry: Donchian breakout in direction of 12h EMA200 trend with volume confirmation
            # Long when price breaks above Donchian high and above 12h EMA200 (bullish bias)
            long_condition = (
                close[i] > highest_high and           # Donchian breakout up
                close[i] > ema200_12h_aligned[i] and  # price above EMA200 (bullish bias)
                vol_confirm                           # volume confirmation
            )
            
            # Short when price breaks below Donchian low and below 12h EMA200 (bearish bias)
            short_condition = (
                close[i] < lowest_low and             # Donchian breakdown down
                close[i] < ema200_12h_aligned[i] and  # price below EMA200 (bearish bias)
                vol_confirm                           # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below 12h EMA200
            if close[i] < ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above 12h EMA200
            if close[i] > ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals