#!/usr/bin/env python3
"""
4h Donchian Breakout + 12h EMA34 Trend + Volume Spike
Hypothesis: Donchian(20) breakouts capture strong momentum moves. 
Filtering by 12h EMA34 trend ensures we trade with the higher timeframe direction, 
reducing false breakouts in choppy markets. Volume confirmation (current volume > 1.5 * 20-period MA) 
adds conviction to breakouts. Designed for 4h timeframe with 75-200 total trades over 4 years 
to balance opportunity and fee drag, working in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need at least 34 periods for EMA34
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = pd.Series(df_12h['close'])
    ema_34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian(20), EMA34, and volume MA
    start_idx = max(34, 20)  # 34 for EMA34 warmup, 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_12h_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Donchian(20) channels: highest high and lowest low of past 20 periods (including current)
        highest_high = np.max(high[i-19:i+1]) if i >= 19 else np.nan
        lowest_low = np.min(low[i-19:i+1]) if i >= 19 else np.nan
        
        if np.isnan(highest_high) or np.isnan(lowest_low):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price relative to 12h EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above Donchian upper channel with volume confirmation in uptrend
            long_breakout = (curr_close > highest_high) and volume_confirm and uptrend
            # Short: price breaks below Donchian lower channel with volume confirmation in downtrend
            short_breakout = (curr_close < lowest_low) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below Donchian lower channel OR EMA34 trend turns down
            if curr_close < lowest_low or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper channel OR EMA34 trend turns up
            if curr_close > highest_high or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0