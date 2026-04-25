#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Donchian20_Breakout_TrendFilter
Hypothesis: 6-hour Donchian(20) breakout aligned with weekly pivot bias and 1d EMA50 trend filter.
Targets 12-25 trades/year by requiring: 1) price breaks 20-period Donchian channel, 2) price is above/below weekly pivot (bull/bear bias), 3) aligned with 1d EMA50 trend.
Uses 6h timeframe to reduce frequency and capture significant moves. Weekly pivot provides structural bias from higher timeframe, avoiding counter-trend entries.
Works in bull/bear by following 1d trend and weekly pivot bias, reducing false breakouts in ranging markets.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w data for weekly pivot (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    weekly_pivot_vals = weekly_pivot.values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_vals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian(20) + 1d EMA50(50) + 1w pivot
    start_idx = max(20, 50) + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Donchian(20) breakout levels (using prior 20 bars, not including current)
        lookback_start = max(0, i - 20)
        lookback_end = i  # exclusive, so we use [lookback_start:lookback_end]
        if lookback_end - lookback_start < 20:
            signals[i] = 0.0
            continue
        highest_high = np.max(high[lookback_start:lookback_end])
        lowest_low = np.min(low[lookback_start:lookback_end])
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Weekly pivot bias: price above/below weekly pivot
        above_pivot = curr_close > weekly_pivot_aligned[i]
        below_pivot = curr_close < weekly_pivot_aligned[i]
        
        if position == 0:
            # Look for entry signals with trend and pivot alignment
            # Long breakout: price breaks above Donchian high with uptrend and above weekly pivot
            long_breakout = (curr_close > highest_high) and uptrend and above_pivot
            # Short breakout: price breaks below Donchian low with downtrend and below weekly pivot
            short_breakout = (curr_close < lowest_low) and downtrend and below_pivot
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price breaks below Donchian low (mean reversion) or trend changes to downtrend
            if curr_close < lowest_low or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above Donchian high (mean reversion) or trend changes to uptrend
            if curr_close > highest_high or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Donchian20_Breakout_TrendFilter"
timeframe = "6h"
leverage = 1.0