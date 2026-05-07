#!/usr/bin/env python3
# 4h_Donchian20_Breakout_TrendVolume
# Hypothesis: On 4h chart, enter long when price breaks above Donchian(20) upper band with volume confirmation and 1d EMA trend filter.
# Enter short when price breaks below Donchian(20) lower band with volume confirmation and 1d EMA trend filter.
# Use Donchian breakouts as a trend-following mechanism that works in both bull and bear markets by capturing strong momentum moves.
# The 1d EMA filter ensures we only trade in the direction of the higher timeframe trend, reducing false signals.
# Volume confirmation adds conviction to breakouts, reducing false breakouts in low-volume environments.
# Designed for low trade frequency (~20-40/year) to minimize fee drag.
# Stoploss via signal=0 when price closes inside the Donchian channel (mean reversion exit).
timeframe = "4h"
name = "4h_Donchian20_Breakout_TrendVolume"
leverage = 1.0

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
    
    # Donchian Channel parameters
    dc_period = 20
    
    # Calculate Donchian Channels
    dc_upper = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    dc_lower = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(dc_period, n):
        # Skip if any critical value is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band + volume spike + 1d EMA uptrend
            if close[i] > dc_upper[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band + volume spike + 1d EMA downtrend
            elif close[i] < dc_lower[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes inside Donchian channel (mean reversion)
            if dc_lower[i] <= close[i] <= dc_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes inside Donchian channel (mean reversion)
            if dc_lower[i] <= close[i] <= dc_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals