#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d Volume Spike + 1w EMA Trend Filter
Hypothesis: On 12h timeframe, Donchian breakouts with volume confirmation and 
weekly EMA trend filter capture strong directional moves while minimizing whipsaws.
Volume spike (2x 20-period average) ensures momentum behind breakouts.
Weekly EMA filter ensures trading only in higher timeframe trend direction.
Target: 50-150 total trades over 4 years with low frequency to minimize fee drag.
Works in both bull (breakouts with trend) and bear (breakouts against trend filtered out).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1dvol_1wema_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume average (once before loop)
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Load 1w data for EMA (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 20-period volume average on daily
    vol_ma_1d = np.full_like(vol_1d, np.nan)
    if len(vol_1d) >= 20:
        vol_ma_1d = np.convolve(vol_1d, np.ones(20)/20, mode='same')
        vol_ma_1d[:10] = np.nan
        vol_ma_1d[-10:] = np.nan
    
    # 50-period EMA on weekly
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        multiplier = 2 / (50 + 1)
        ema_1w[0] = close_1w[0]
        for i in range(1, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Align to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 50)  # For Donchian and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Volume filter: current volume > 2x 20-period daily average
        volume_filter = volume[i] > (vol_ma_1d_aligned[i] * 2.0) if not np.isnan(vol_ma_1d_aligned[i]) else False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR price below weekly EMA
            if close[i] < lowest_low or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR price above weekly EMA
            if close[i] > highest_high or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly EMA trend filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Only go long if price above weekly EMA, short if below
            weekly_uptrend = close[i] > ema_1w_aligned[i]
            weekly_downtrend = close[i] < ema_1w_aligned[i]
            
            if i >= 20 and bull_breakout and volume_filter and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            elif i >= 20 and bear_breakout and volume_filter and weekly_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals