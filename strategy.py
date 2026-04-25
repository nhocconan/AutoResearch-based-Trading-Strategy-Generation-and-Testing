#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: 6h Donchian(20) breakouts in the direction of 1-week EMA50 trend with volume spike confirmation capture institutional order flow across market regimes. Weekly trend filter ensures we only trade with the dominant higher-timeframe momentum, reducing false breakouts in chop. Volume spike confirms participation. Target: 12-30 trades/year (50-120 over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for EMA50 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1-week EMA50 trend filter
    ema_50_1w = calculate_ema(df_1w['close'].values, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h Donchian channels (20-period)
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20) + volume MA (20) + 1w EMA (50)
    start_idx = max(donchian_window, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + 1w EMA50 trend alignment
            long_breakout = curr_high > upper[i]
            short_breakout = curr_low < lower[i]
            
            long_entry = long_breakout and volume_spike[i] and (curr_close > ema_50_1w_aligned[i])
            short_entry = short_breakout and volume_spike[i] and (curr_close < ema_50_1w_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Donchian lower (failed breakout) or trend turns bearish
            if curr_close < lower[i] or curr_close < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Donchian upper (failed breakout) or trend turns bullish
            if curr_close > upper[i] or curr_close > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0