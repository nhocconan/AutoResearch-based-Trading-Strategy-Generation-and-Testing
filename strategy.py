#!/usr/bin/env python3
"""
1d_TurtleChannel_1wTrendFilter_Volume
Hypothesis: Use a 20-bar Donchian channel on daily closes for breakout signals,
filtered by 1-week EMA trend direction and volume expansion. Exit when price
reverses to the opposite channel boundary or trend changes. Designed for low
trade frequency (~15-25/year) with controlled risk via trend alignment.
Works in bull markets via upside breakouts and in bear markets via downside
breakouts, using the higher timeframe trend to avoid counter-trend trades.
"""

name = "1d_TurtleChannel_1wTrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on weekly closes for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Donchian channel: 20-period high/low of closes
    # Use pandas rolling for clarity and proper min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-day average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian and weekly EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_1d[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Trend filter: price above/below weekly EMA20
        trend_up = close[i] > ema_20_1d[i]
        trend_down = close[i] < ema_20_1d[i]
        
        if position == 0:
            # Long: break above Donchian high + uptrend + volume
            if breakout_up and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + downtrend + volume
            elif breakout_down and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend turns down
            if close[i] < donchian_low[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend turns up
            if close[i] > donchian_high[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals