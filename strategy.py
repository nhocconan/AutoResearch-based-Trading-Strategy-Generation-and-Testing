#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_1dTrendFilter_v1
Hypothesis: Combines weekly pivot structure with 6h Donchian breakouts filtered by 1d EMA trend.
In bull markets: weekly R1/R2 act as support for continuation longs above weekly PP.
In bear markets: weekly S1/S2 act as resistance for continuation shorts below weekly PP.
Weekly pivot provides meaningful structure that adapts to changing volatility.
Donchian(20) breakout captures momentum, filtered by 1d EMA50 for trend alignment.
Targets 12-30 trades/year to minimize fee drag in 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot levels (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot calculations (standard floor trader pivots)
    PP_1w = (high_1w + low_1w + close_1w) / 3.0
    R1_1w = 2 * PP_1w - low_1w
    S1_1w = 2 * PP_1w - high_1w
    R2_1w = PP_1w + (high_1w - low_1w)
    S2_1w = PP_1w - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    PP_1w_aligned = align_htf_to_ltf(prices, df_1w, PP_1w)
    R1_1w_aligned = align_htf_to_ltf(prices, df_1w, R1_1w)
    S1_1w_aligned = align_htf_to_ltf(prices, df_1w, S1_1w)
    R2_1w_aligned = align_htf_to_ltf(prices, df_1w, R2_1w)
    S2_1w_aligned = align_htf_to_ltf(prices, df_1w, S2_1w)
    
    # Calculate Donchian channels (20-period) on 6h
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume filter: above average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    volume_filter = volume > vol_avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(donchian_window, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(PP_1w_aligned[i]) or 
            np.isnan(R1_1w_aligned[i]) or np.isnan(S1_1w_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for Donchian breakout with weekly pivot bias and trend filter
            # Long: break above Donchian high + above weekly PP + price above 1d EMA50 + volume
            long_breakout = close_val > highest_high[i]
            long_pivot_bias = close_val > PP_1w_aligned[i]  # Above weekly pivot = bullish bias
            long_trend = close_val > ema_50_1d_aligned[i]
            long_volume = volume_filter[i]
            
            # Short: break below Donchian low + below weekly PP + price below 1d EMA50 + volume
            short_breakout = close_val < lowest_low[i]
            short_pivot_bias = close_val < PP_1w_aligned[i]  # Below weekly pivot = bearish bias
            short_trend = close_val < ema_50_1d_aligned[i]
            short_volume = volume_filter[i]
            
            if long_breakout and long_pivot_bias and long_trend and long_volume:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_breakout and short_pivot_bias and short_trend and short_volume:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price breaks below weekly S1 or Donchian low
            exit_condition = (close_val < S1_1w_aligned[i]) or (close_val < lowest_low[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above weekly R1 or Donchian high
            exit_condition = (close_val > R1_1w_aligned[i]) or (close_val > highest_high[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0