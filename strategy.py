#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Long when price breaks above Donchian upper band AND weekly pivot > previous weekly pivot AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower band AND weekly pivot < previous weekly pivot AND volume > 1.5x 20-period average
# Uses 6h timeframe to reduce trade frequency vs lower timeframes
# Weekly pivot filter ensures we only trade in direction of weekly trend
# Discrete position sizing (0.25) to minimize fee drag
# Target: 12-25 trades/year on 6h timeframe (~50-100 total over 4 years)

name = "6h_Donchian_20_Breakout_WeeklyPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 30, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        curr_weekly_pivot = weekly_pivot_aligned[i]
        curr_vol_ma_20 = vol_ma_20[i]
        
        # Skip if weekly pivot is not available
        if np.isnan(curr_weekly_pivot):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation
        vol_spike = curr_volume > 1.5 * curr_vol_ma_20 if curr_vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop
            if curr_close < stop_price:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop
            if curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band AND weekly pivot > previous weekly pivot AND volume spike
            if (curr_close > highest_20[i] and 
                curr_weekly_pivot > weekly_pivot_aligned[i-1] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below Donchian lower band AND weekly pivot < previous weekly pivot AND volume spike
            elif (curr_close < lowest_20[i] and 
                  curr_weekly_pivot < weekly_pivot_aligned[i-1] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals