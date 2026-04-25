#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Donchian breakouts capture strong momentum moves. 
Combined with 1d EMA34 trend filter and volume confirmation, this should work in both bull and bear markets by only taking breakouts in the direction of the higher timeframe trend.
Targets 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 24-period volume MA for 12h volume confirmation (24 * 12h = 12d ~ 2 weeks)
    vol_ma_24_12h = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24_12h[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian(20), EMA, and volume MA
    start_idx = max(20, 24, 34)  # 20 for Donchian, 24 for volume MA, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_24_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ma_12h = vol_ma_24_12h[i]
        
        # Donchian(20) breakout levels
        highest_high = np.max(high[i-19:i+1])   # 20-period high including current
        lowest_low = np.min(low[i-19:i+1])      # 20-period low including current
        
        # Volume confirmation: current 12h volume > 2.0 * 24-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_close > highest_high and 
                         curr_close > ema_trend and volume_confirm)
            # Short: price breaks below Donchian low AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_close < lowest_low and 
                          curr_close < ema_trend and volume_confirm)
            
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
            # Exit: price falls below Donchian low OR price crosses below EMA34
            if (curr_close < lowest_low or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian high OR price crosses above EMA34
            if (curr_close > highest_high or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0