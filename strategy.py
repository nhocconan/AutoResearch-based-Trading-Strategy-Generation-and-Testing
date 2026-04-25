#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_Trend_VolumeConfirm
Hypothesis: 6h Donchian(20) breakout with 1w pivot direction (long above weekly pivot, short below) and volume confirmation (>1.5x 20-period average). 
Weekly pivot provides institutional reference; Donchian captures breakouts; volume filters weak moves. 
Designed for low turnover (12-25 trades/year) and robustness in bull/bear via trend alignment and institutional filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1w data for weekly pivot (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # 1d data for Donchian(20) (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d Donchian20 (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation
            # Long breakout: price breaks above Donchian high AND above weekly pivot
            long_breakout = (curr_close > donchian_high_aligned[i]) and (curr_close > weekly_pivot_aligned[i]) and volume_confirm[i]
            # Short breakout: price breaks below Donchian low AND below weekly pivot
            short_breakout = (curr_close < donchian_low_aligned[i]) and (curr_close < weekly_pivot_aligned[i]) and volume_confirm[i]
            
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
            # Stoploss: 2.0 * ATR below entry (using 6h ATR)
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks below Donchian low or weekly pivot
            elif curr_close < donchian_low_aligned[i] or curr_close < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Calculate 6h ATR (same as above)
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above Donchian high or weekly pivot
            elif curr_close > donchian_high_aligned[i] or curr_close > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0