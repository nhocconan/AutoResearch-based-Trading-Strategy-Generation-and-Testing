#!/usr/bin/env python3
"""
Hypothesis: Daily timeframe strategy using weekly pivot point breakout with volume confirmation and weekly trend filter.
Enters long when price breaks above weekly R1 with volume > 1.5x daily average and weekly close above weekly open (bullish weekly candle).
Enters short when price breaks below weekly S1 with volume > 1.5x daily average and weekly close below weekly open (bearish weekly candle).
Exits when price returns to weekly pivot level. Uses weekly timeframe for structure and trend, daily for execution.
Designed to work in both bull and bear markets by requiring alignment with weekly candle direction and volume confirmation.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Support and resistance levels
    S1 = (2 * pivot) - high_1w
    R1 = (2 * pivot) - low_1w
    
    # Align weekly pivot levels to daily timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Weekly bullish/bearish candle: close > open (bullish), close < open (bearish)
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivot levels, volume MA
    start_idx = max(5, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current daily price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Current weekly levels
        S1_now = S1_aligned[i]
        R1_now = R1_aligned[i]
        pivot_now = pivot_aligned[i]
        
        # Weekly candle direction
        is_bullish_weekly = weekly_bullish_aligned[i] == 1
        is_bearish_weekly = weekly_bearish_aligned[i] == 1
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Weekly pivot breakout with volume and weekly candle alignment
        if position == 0:
            # Long: price breaks above R1 with volume + bullish weekly candle
            if price_now > R1_now and vol_filter and is_bullish_weekly:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with volume + bearish weekly candle
            elif price_now < S1_now and vol_filter and is_bearish_weekly:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly pivot
            if price_now <= pivot_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to weekly pivot
            if price_now >= pivot_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyPivotBreakout_Volume_WeeklyCandle"
timeframe = "1d"
leverage = 1.0