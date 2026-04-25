#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Confluence
Hypothesis: 6-hour Donchian(20) breakout with weekly pivot direction and volume confirmation.
Targets 12-37 trades/year by requiring: 1) price breaks 6h Donchian(20) channels,
2) aligned with weekly Camarilla H4/L4 bias (bullish above H4, bearish below L4),
3) volume > 1.8x 20-period average. Weekly pivot provides structural bias that works
in both bull and bear markets by identifying key institutional levels. Donchian(20)
captures breakouts with clear risk definition. Volume confirmation reduces false
breakouts in low-participation environments.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1w data for weekly Camarilla pivot bias (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    prev_close_w = df_1w['close'].shift(1).values
    prev_high_w = df_1w['high'].shift(1).values
    prev_low_w = df_1w['low'].shift(1).values
    prev_range_w = prev_high_w - prev_low_w
    
    # Weekly Camarilla H4 and L4 levels (strong bias levels)
    H4_w = prev_close_w + 1.1 * prev_range_w * (1.0)
    L4_w = prev_close_w - 1.1 * prev_range_w * (1.0)
    
    # Align weekly levels to 6h timeframe
    H4_w_aligned = align_htf_to_ltf(prices, df_1w, H4_w)
    L4_w_aligned = align_htf_to_ltf(prices, df_1w, L4_w)
    
    # Weekly bias: bullish above H4, bearish below L4, neutral between
    weekly_bullish = prev_close_w > H4_w  # Weekly close above H4 = bullish bias
    weekly_bearish = prev_close_w < L4_w  # Weekly close below L4 = bearish bias
    
    # Align weekly bias to 6h (shifted by 1 week for completed bar)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # 6h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian(20) and weekly data alignment
    start_idx = max(lookback, 35)  # 35 for weekly data safety
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]) or
            np.isnan(H4_w_aligned[i]) or np.isnan(L4_w_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation and weekly bias
            # Long breakout: price breaks above Donchian HIGH with bullish weekly bias and volume
            long_breakout = (curr_close > highest_high[i]) and weekly_bullish_aligned[i] > 0.5 and volume_confirm[i]
            # Short breakout: price breaks below Donchian LOW with bearish weekly bias and volume
            short_breakout = (curr_close < lowest_low[i]) and weekly_bearish_aligned[i] > 0.5 and volume_confirm[i]
            
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
            # Long position: exit if price breaks below Donchian LOW or weekly bias turns bearish
            if curr_close < lowest_low[i] or weekly_bearish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above Donchian HIGH or weekly bias turns bullish
            if curr_close > highest_high[i] or weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Confluence"
timeframe = "6h"
leverage = 1.0