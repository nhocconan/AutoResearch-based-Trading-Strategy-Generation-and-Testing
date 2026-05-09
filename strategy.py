#!/usr/bin/env python3
# Hypothesis: 4h timeframe with daily donchian breakout + volume confirmation + 1w EMA trend filter.
# Uses daily donchian breakout (20) for structural breakouts and weekly EMA50 for trend filter.
# Daily breakout provides clear entry/exit signals while weekly trend filter reduces whipsaw.
# Volume confirmation ensures breakouts have conviction. Target: 75-200 total trades over 4 years (19-50/year) with size 0.25.

name = "4h_Donchian_Breakout_1wEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate daily Donchian channels (20-day high/low) from previous day
    prev_high = np.roll(high, 6)   # 6 bars = 1 day * 6 bars per 4h
    prev_low = np.roll(low, 6)
    prev_high[:6] = np.nan
    prev_low[:6] = np.nan
    
    # Donchian breakout: price breaks above/below previous day's high/low
    breakout_up = close > prev_high
    breakout_down = close < prev_low
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    trend_up = close > ema_50_1w_aligned
    trend_down = close < ema_50_1w_aligned
    
    # Volume filter: current volume > 1.5x 30-period average volume
    avg_volume = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above previous day's high + 1w uptrend + volume filter
            if breakout_up[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below previous day's low + 1w downtrend + volume filter
            elif breakout_down[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to previous day's low or trend reversal
            if close[i] <= prev_low[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to previous day's high or trend reversal
            if close[i] >= prev_high[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals