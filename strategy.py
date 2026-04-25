#!/usr/bin/env python3
"""
4h Donchian Breakout with 12h EMA50 Trend and Volume Spike Confirmation
Hypothesis: Donchian(20) breakouts capture strong momentum. In bull/bear regimes,
trading only in the direction of 12h EMA50 reduces whipsaw. Volume spike confirms
institutional participation. Targets 75-200 total trades over 4 years (19-50/year).
Works in bull via breakouts, in bear via short breakdowns with trend filter.
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
    
    # Get 12h data for EMA50 trend (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 12h close for trend
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period Donchian channels on 4h
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-19:i+1])
        lowest_20[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for 4h volume confirmation
    vol_ma_20_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_12h_aligned[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        vol_ma_4h = vol_ma_20_4h[i]
        
        # Volume confirmation: current 4h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_4h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian AND price > EMA50 (uptrend) AND volume confirmation
            long_entry = (curr_close > upper_channel and 
                         curr_close > ema_trend and volume_confirm)
            # Short: price breaks below lower Donchian AND price < EMA50 (downtrend) AND volume confirmation
            short_entry = (curr_close < lower_channel and 
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
            # Exit: price falls below lower Donchian OR price crosses below EMA50
            if (curr_close < lower_channel or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above upper Donchian OR price crosses above EMA50
            if (curr_close > upper_channel or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0