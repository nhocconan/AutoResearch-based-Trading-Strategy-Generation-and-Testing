#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Long when: price breaks above Donchian(20) high AND weekly pivot trend is bullish AND volume > 1.5x average
# Short when: price breaks below Donchian(20) low AND weekly pivot trend is bearish AND volume > 1.5x average
# Exit when price returns to Donchian(20) midline or opposite breakout occurs.
# Uses weekly pivot for trend filter to avoid counter-trend trades, volume to confirm breakout strength.
# Designed for low trade frequency (target: 15-30/year) to minimize fee drag in 6h timeframe.
name = "6h_Donchian20_WeeklyPivot_Volume"
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
    
    # Donchian(20) channels
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    midline = (highest_high + lowest_low) / 2.0
    
    # Weekly pivot points (using weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot: P = (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Trend: bullish if price > pivot, bearish if price < pivot
    weekly_bullish = weekly_pivot > 0  # placeholder, will be replaced properly
    weekly_bearish = weekly_pivot > 0   # placeholder, will be replaced properly
    
    # Actually determine trend based on price vs pivot
    # We need to compare current price to weekly pivot
    # Since we can't easily get current weekly pivot in loop, we'll use the pivot value
    # and determine trend based on whether price is above/below it
    # We'll calculate this properly by getting the weekly pivot value and comparing
    
    # Get weekly pivot series and align to 6h
    weekly_pivot_series = weekly_pivot
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_series)
    
    # Determine weekly trend: bullish if close > weekly pivot, bearish if close < weekly pivot
    weekly_trend_bullish = close > weekly_pivot_aligned
    weekly_trend_bearish = close < weekly_pivot_aligned
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_surge = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback-1, 19)  # Need enough data for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian high AND weekly bullish AND volume surge
            long_breakout = close[i] > highest_high[i]
            # Short breakout: price < Donchian low AND weekly bearish AND volume surge
            short_breakout = close[i] < lowest_low[i]
            
            if long_breakout and weekly_trend_bullish[i] and volume_surge[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and weekly_trend_bearish[i] and volume_surge[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < Donchian midline OR short breakout occurs
            if close[i] < midline[i] or (close[i] < lowest_low[i] and weekly_trend_bearish[i] and volume_surge[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > Donchian midline OR long breakout occurs
            if close[i] > midline[i] or (close[i] > highest_high[i] and weekly_trend_bullish[i] and volume_surge[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals