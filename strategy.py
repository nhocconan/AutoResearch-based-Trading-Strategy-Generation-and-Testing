#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_v3
Hypothesis: Trade 6h Donchian(20) breakouts in direction of weekly pivot trend.
Weekly pivot (PP) calculated from prior week OHLC. Price above PP = bullish bias (long breakouts),
price below PP = bearish bias (short breakouts). Uses volume confirmation to avoid false breakouts.
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
Works in bull markets via long breakouts above weekly PP, in bear markets via short breakdowns below weekly PP.
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
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week
    # PP = (H + L + C) / 3
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_week_high = np.where(np.isnan(prev_week_high), df_1w['high'].values, prev_week_high)
    prev_week_low = np.where(np.isnan(prev_week_low), df_1w['low'].values, prev_week_low)
    prev_week_close = np.where(np.isnan(prev_week_close), df_1w['close'].values, prev_week_close)
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    
    # Align weekly pivot to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) on 6h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1w data, Donchian(20), volume MA(20)
    start_idx = max(20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        bullish_bias = close_val > weekly_pivot_aligned[i]   # price above weekly PP
        bearish_bias = close_val < weekly_pivot_aligned[i]   # price below weekly PP
        
        if position == 0:
            # Long: price breaks above Donchian HIGH AND volume confirm AND bullish bias
            long_signal = (close_val > donchian_high[i]) and vol_conf and bullish_bias
            
            # Short: price breaks below Donchian LOW AND volume confirm AND bearish bias
            short_signal = (close_val < donchian_low[i]) and vol_conf and bearish_bias
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price drops below Donchian LOW (failed breakout) OR bullish bias flips
            if (close_val < donchian_low[i]) or (not bullish_bias):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian HIGH (failed breakdown) OR bearish bias flips
            if (close_val > donchian_high[i]) or (not bearish_bias):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_v3"
timeframe = "6h"
leverage = 1.0