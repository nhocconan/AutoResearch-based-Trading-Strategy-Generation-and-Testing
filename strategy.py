#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d daily pivot context and volume confirmation
# Daily pivot levels provide institutional support/resistance
# Donchian breakout captures momentum in direction of pivot bias
# Volume > 2x average confirms institutional participation
# Works in bull/bear as pivot bias adapts to trend
# Target: 12-37 trades/year per symbol (48-148 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for daily pivot and trend context
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points from prior day (using 1d data)
    lookback = 1
    if len(df_1d) < lookback:
        return np.zeros(n)
    
    # Get prior day's OHLC (excluding current incomplete day)
    prev_day_high = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    prev_day_low = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    prev_day_close = pd.Series(df_1d['close']).rolling(window=lookback, min_periods=lookback).last().shift(1).values
    
    # Daily pivot calculation (standard floor trader pivot)
    daily_pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    daily_r1 = 2 * daily_pivot - prev_day_low
    daily_s1 = 2 * daily_pivot - prev_day_high
    daily_r2 = daily_pivot + (prev_day_high - prev_day_low)
    daily_s2 = daily_pivot - (prev_day_high - prev_day_low)
    
    # Align daily pivot levels to 12h timeframe
    daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    daily_r2_aligned = align_htf_to_ltf(prices, df_1d, daily_r2)
    daily_s2_aligned = align_htf_to_ltf(prices, df_1d, daily_s2)
    
    # Donchian channel (20 periods) on 12h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 2x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(daily_pivot_aligned[i]) or
            np.isnan(daily_r1_aligned[i]) or
            np.isnan(daily_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Pivot bias: price relative to daily pivot
        above_pivot = close[i] > daily_pivot_aligned[i]
        below_pivot = close[i] < daily_pivot_aligned[i]
        
        # Volume confirmation: current volume > 2x average
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + above daily pivot + volume
            if (close[i] > dc_upper[i] and 
                above_pivot and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + below daily pivot + volume
            elif (close[i] < dc_lower[i] and 
                  below_pivot and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to daily pivot or breaks below S1
            if close[i] < daily_pivot_aligned[i] or close[i] < daily_s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to daily pivot or breaks above R1
            if close[i] > daily_pivot_aligned[i] or close[i] > daily_r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_DailyPivot_Donchian_Volume_v1"
timeframe = "12h"
leverage = 1.0