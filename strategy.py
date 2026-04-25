#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_WeeklyPivotDir_v1
Hypothesis: Trade 6h Donchian(20) breakouts with 1d EMA50 trend filter and weekly pivot direction filter.
In bull markets (price > 1d EMA50): long when price breaks above 6h Donchian(20) high AND weekly pivot is bullish.
In bear markets (price < 1d EMA50): short when price breaks below 6h Donchian(20) low AND weekly pivot is bearish.
Weekly pivot direction based on price vs weekly VWAP: bullish if close > weekly VWAP, bearish if close < weekly VWAP.
Exit on opposite Donchian touch or trend reversal. Position size: 0.25.
Target: 75-150 total trades over 4 years = 19-38/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly VWAP for pivot direction
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    volume_1w = df_1w['volume'].values
    vwap_1w = np.cumsum(typical_price_1w * volume_1w) / np.cumsum(volume_1w)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w, additional_delay_bars=0)
    
    # Calculate 6h Donchian(20) channels
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vwap_1w_aligned[i]) or
            np.isnan(high_ma_20[i]) or
            np.isnan(low_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above 1d EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Determine weekly pivot direction (bullish = price above weekly VWAP)
        weekly_bullish = close[i] > vwap_1w_aligned[i]
        weekly_bearish = close[i] < vwap_1w_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above 6h Donchian(20) high + 1d uptrend + weekly bullish pivot
            long_setup = (close[i] > high_ma_20[i]) and htf_1d_bullish and weekly_bullish
            
            # Short setup: price breaks below 6h Donchian(20) low + 1d downtrend + weekly bearish pivot
            short_setup = (close[i] < low_ma_20[i]) and htf_1d_bearish and weekly_bearish
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches 6h Donchian(20) low (stop) OR 1d trend turns bearish
            if (close[i] <= low_ma_20[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches 6h Donchian(20) high (stop) OR 1d trend turns bullish
            if (close[i] >= high_ma_20[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_WeeklyPivotDir_v1"
timeframe = "6h"
leverage = 1.0