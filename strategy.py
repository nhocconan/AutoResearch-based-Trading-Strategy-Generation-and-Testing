#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: On 6h timeframe, Donchian channel breakouts aligned with weekly pivot
direction (from weekly Camarilla R4/S4 levels) and volume confirmation capture
strong momentum moves. Weekly pivot provides longer-term bias, reducing false
breakouts in choppy markets. Works in bull/bear via weekly trend filter.
Target: 12-25 trades/year on 6h to avoid fee drag.
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
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (R4/S4 for strong breakout bias)
    # Typical price = (high + low + close) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    range_1w = df_1w['high'] - df_1w['low']
    
    # Weekly Camarilla levels: R4/S4 are the strongest breakout levels
    # R4 = close + (high - low) * 1.1 / 2
    # S4 = close - (high - low) * 1.1 / 2
    r4 = df_1w['close'] + range_1w * 1.1 / 2
    s4 = df_1w['close'] - range_1w * 1.1 / 2
    
    # Align the weekly pivot levels to LTF (they represent the previous week's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4.values)
    
    # Load daily data for volume confirmation (more stable than intraday)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily volume MA for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 6h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and volume MA
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = curr_volume > (vol_ma_1d_aligned[i] * 2.0)
        
        # Weekly bias: price relative to weekly R4/S4
        weekly_bullish = curr_close > r4_aligned[i]  # Above weekly R4 = strong bullish bias
        weekly_bearish = curr_close < s4_aligned[i]  # Below weekly S4 = strong bearish bias
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + weekly bias + volume spike
            # Long: break above Donchian high AND weekly bullish bias AND volume spike
            long_entry = (curr_high > highest_high[i]) and weekly_bullish and vol_spike
            # Short: break below Donchian low AND weekly bearish bias AND volume spike
            short_entry = (curr_low < lowest_low[i]) and weekly_bearish and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price re-enters Donchian channel OR weekly bias turns bearish
            if (curr_close < highest_high[i] and curr_close > lowest_low[i]) or not weekly_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price re-enters Donchian channel OR weekly bias turns bullish
            if (curr_close < highest_high[i] and curr_close > lowest_low[i]) or not weekly_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotR4S4_VolumeSpike"
timeframe = "6h"
leverage = 1.0