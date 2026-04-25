#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDir_VolumeConfirm
Hypothesis: On 6h timeframe, use Donchian(20) breakouts filtered by weekly pivot direction (from 1w HTF) and volume confirmation. Weekly pivot direction provides structural bias: price above weekly pivot = bullish bias (long breakouts), below = bearish bias (short breakouts). Volume confirmation reduces false breakouts. Designed for low trade frequency (12-37/year) to minimize fee drag. Works in bull markets via long breakouts and bear markets via short breakouts, with weekly pivot adapting to longer-term trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for weekly pivot (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian(20) and volume MA
    start_idx = lookback  # 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Look for entry signals
            # Long breakout: price breaks above Donchian high AND above weekly pivot (bullish bias) AND volume spike
            long_breakout = (curr_high > highest_high[i]) and (curr_close > weekly_pivot_aligned[i]) and volume_spike[i]
            
            # Short breakout: price breaks below Donchian low AND below weekly pivot (bearish bias) AND volume spike
            short_breakout = (curr_low < lowest_low[i]) and (curr_close < weekly_pivot_aligned[i]) and volume_spike[i]
            
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
            # Long position: exit when price breaks below Donchian low (contrary breakout) or volume spike reversal
            if curr_low < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above Donchian high (contrary breakout) or volume spike reversal
            if curr_high > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotDir_VolumeConfirm"
timeframe = "6h"
leverage = 1.0