#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture momentum in the direction of the 1d EMA34 trend.
Volume spike confirms institutional participation. Designed for 4h timeframe with tight entry
conditions to achieve 75-200 total trades over 4 years (19-50/year). Works in bull (breakouts
above upper channel in uptrend) and bear (breakouts below lower channel in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) channels on primary timeframe (4h)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA, volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        upper_channel = high_roll[i]
        lower_channel = low_roll[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian channel AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_high > upper_channel) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below lower Donchian channel AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_low < lower_channel) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below lower Donchian channel OR price crosses below EMA (trend change)
            if (curr_low < lower_channel) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above upper Donchian channel OR price crosses above EMA (trend change)
            if (curr_high > upper_channel) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0