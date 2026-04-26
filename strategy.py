#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter (from 1w), 1d EMA50 trend filter, and volume spike confirmation. Weekly pivot provides structural bias (bull/bear/range) that works in both bull (breakouts with trend) and bear (fade at extremes with volume exhaustion). Volume spike ensures institutional participation. Discrete sizing (0.25) targets 12-30 trades/year to minimize fee drag in ranging/bear markets like 2025+.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter: volume > 2.5 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    # Calculate Donchian channels (20-period) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot from previous 1w bar
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_week_high = np.where(np.isnan(prev_week_high), df_1w['high'].values, prev_week_high)
    prev_week_low = np.where(np.isnan(prev_week_low), df_1w['low'].values, prev_week_low)
    prev_week_close = np.where(np.isnan(prev_week_close), df_1w['close'].values, prev_week_close)
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA(50), volume MA, Donchian, ATR
    start_idx = max(50, 50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_1d_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_1d_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly pivot AND 1d trend up AND volume spike
            long_signal = (close_val > highest_high[i]) and (close_val > weekly_pivot_aligned[i]) and trend_1d_up and vol_spike
            
            # Short: price breaks below Donchian low AND below weekly pivot AND 1d trend down AND volume spike
            short_signal = (close_val < lowest_low[i]) and (close_val < weekly_pivot_aligned[i]) and trend_1d_down and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend flips down OR price hits ATR stoploss (2.0x) OR price crosses below weekly pivot
            if (not trend_1d_up) or (close_val < entry_price - 2.0 * atr[i]) or (close_val < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss (2.0x) OR price crosses above weekly pivot
            if (not trend_1d_down) or (close_val > entry_price + 2.0 * atr[i]) or (close_val > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0