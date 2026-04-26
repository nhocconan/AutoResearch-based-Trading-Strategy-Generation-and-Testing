#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_HTFVolume_v1
Hypothesis: On 6h timeframe, trade Donchian(20) breakouts only when aligned with weekly pivot direction (bullish if price > weekly pivot, bearish if < weekly pivot) and HTF volume confirmation (1d volume > 1.5x 20-period median). Weekly pivot adds structural bias from higher timeframe, reducing false breakouts in choppy markets. Volume filter ensures conviction. Designed to work in both bull and bear markets by only trading with the weekly trend. Targets 12-30 trades/year via tight entry conditions requiring confluence of breakout, weekly pivot direction, and volume.
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
    
    # Get 1d data for HTF trend (EMA34) and volume
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d volume for confirmation
    vol_1d = df_1d['volume'].values
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot from previous weekly OHLC
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    
    # Align HTF indicators to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter: 1d volume > 1.5x 20-period median for conviction
    vol_median = pd.Series(vol_1d_aligned).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), EMA(34) 1d, volume median (20)
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_median_val = vol_median[i]
        highest_20_val = highest_20[i]
        lowest_20_val = lowest_20[i]
        
        # Trend filter from 1d EMA34: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1d_val
        downtrend = close_val < ema_34_1d_val
        
        # Weekly pivot direction: price > weekly pivot (bullish bias) or < weekly pivot (bearish bias)
        bullish_bias = close_val > weekly_pivot_val
        bearish_bias = close_val < weekly_pivot_val
        
        # Volume spike filter: only trade in above-average volume environments
        volume_spike = vol_1d_aligned[i] > 1.5 * vol_median_val
        
        if position == 0:
            # Long: break above Donchian high with volume spike, uptrend, and bullish bias
            long_signal = (close_val > highest_20_val) and \
                          volume_spike and \
                          uptrend and \
                          bullish_bias
            
            # Short: break below Donchian low with volume spike, downtrend, and bearish bias
            short_signal = (close_val < lowest_20_val) and \
                           volume_spike and \
                           downtrend and \
                           bearish_bias
            
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
            # Exit: close below Donchian low (breakdown) or loss of weekly bullish bias
            if close_val < lowest_20_val or not bullish_bias:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close above Donchian high (breakout) or loss of weekly bearish bias
            if close_val > highest_20_val or not bearish_bias:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_HTFVolume_v1"
timeframe = "6h"
leverage = 1.0