#!/usr/bin/env python3
"""
1h Donchian Channel Breakout + 4h EMA Trend + Volume Spike + Session Filter
Hypothesis: Donchian(20) breakouts on 1h capture intraday momentum. 
4h EMA50 filter ensures we trade with the higher timeframe trend to avoid whipsaws.
Volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise.
Designed for low trade frequency: ~20-40 trades/year per symbol. Works in bull/bear via trend filter.
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
    
    # Get 4h data for EMA trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1h Donchian(20) and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if EMA not ready
        if np.isnan(ema_50_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1h Donchian(20): highest high and lowest low of past 20 bars (excluding current)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume spike: current volume > 2.0 * 20-period average
        vol_ma_20 = np.mean(volume[i-20:i])
        volume_spike = volume[i] > 2.0 * vol_ma_20
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 4h EMA50 uptrend AND volume spike
            long_condition = (curr_close > highest_high) and (curr_close > ema_50_4h_aligned[i]) and volume_spike
            # Short: price breaks below Donchian lower AND 4h EMA50 downtrend AND volume spike
            short_condition = (curr_close < lowest_low) and (curr_close < ema_50_4h_aligned[i]) and volume_spike
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian lower or 4h EMA50 turns down
            if curr_close < lowest_low or curr_close < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns above Donchian upper or 4h EMA50 turns up
            if curr_close > highest_high or curr_close > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_Breakout_4hEMA50_Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0