#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with weekly trend filter and volume confirmation.
# Uses Donchian(20) on 6h for breakout signals, filtered by weekly trend (price above/below weekly SMA20).
# Volume must be above 1.5x average to confirm breakout strength.
# Designed for low-frequency, high-conviction trades (~15-25/year) to minimize fee drag.
# Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMA20 for trend filter
    sma20_1w = np.full_like(close_1w, np.nan)
    for i in range(19, len(close_1w)):
        sma20_1w[i] = np.mean(close_1w[i-19:i+1])
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Align weekly SMA20 to 6h timeframe
    sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(sma20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Donchian breakout conditions
        breakout_up = price > highest_high[i]
        breakdown_down = price < lowest_low[i]
        
        # Weekly trend filter
        weekly_uptrend = price > sma20_1w_aligned[i]
        weekly_downtrend = price < sma20_1w_aligned[i]
        
        if position == 0:
            # Long: upward breakout with volume and weekly uptrend
            if breakout_up and vol_filter and weekly_uptrend:
                signals[i] = size
                position = 1
            # Short: downward breakdown with volume and weekly downtrend
            elif breakdown_down and vol_filter and weekly_downtrend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly SMA20 or breakdown occurs
            if price < sma20_1w_aligned[i] or breakdown_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly SMA20 or breakout occurs
            if price > sma20_1w_aligned[i] or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0