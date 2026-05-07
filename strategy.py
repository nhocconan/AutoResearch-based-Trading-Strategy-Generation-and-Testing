#!/usr/bin/env python3
# 12h_Donchian20_Breakout_Volume_Trend_1d
# Hypothesis: On 12h chart, enter long when price breaks above 20-period Donchian upper band with volume confirmation and 1d EMA trend,
# enter short when price breaks below 20-period Donchian lower band with volume confirmation and 1d EMA trend.
# Uses 1d EMA for trend filter to avoid counter-trend trades. Designed for low trade frequency (~15-30/year) to minimize fee drift.
# Donchian channels capture breakouts, volume confirms strength, and EMA filter ensures trend alignment.
# Works in both bull and bear markets by filtering trades with higher timeframe trend.
timeframe = "12h"
name = "12h_Donchian20_Breakout_Volume_Trend_1d"
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
    
    # Load 1h data for EMA trend filter (using 1h as proxy for trend since 1d may be too sparse)
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    ema_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(dc_period, n):
        # Skip if any critical value is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema_1h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band + volume spike + price above 1h EMA
            if close[i] > dc_upper[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_1h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band + volume spike + price below 1h EMA
            elif close[i] < dc_lower[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_1h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below Donchian lower band (stoploss)
            if close[i] < dc_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above Donchian upper band (stoploss)
            if close[i] > dc_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals