#!/usr/bin/env python3

"""
Hypothesis: 12-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
Breakouts above/below Donchian(20) channels capture momentum moves. Daily trend filter
avoids counter-trend trades by only allowing long in bullish daily trend and short
in bearish daily trend. Volume spikes confirm breakout strength. This structure
works in both bull and bear markets by adapting to the daily trend.
Target: 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, window=20):
    """Calculate Donchian channel upper and lower bands"""
    upper = pd.Series(high).rolling(window=window, min_periods=window).max()
    lower = pd.Series(low).rolling(window=window, min_periods=window).min()
    return upper.values, lower.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Determine daily trend: price above/below EMA34
    bullish_trend = close_1d > ema_34_1d
    bearish_trend = close_1d < ema_34_1d
    
    # Align daily trend to 12h timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # Calculate 12h Donchian channel (20-period)
    upper_12h, lower_12h = calculate_donchian(high, low, 20)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above upper band, bullish daily trend, volume spike
            if (close[i] > upper_12h[i] and
                bullish_aligned[i] > 0.5 and
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.30
                position = 1
            # Short: breakout below lower band, bearish daily trend, volume spike
            elif (close[i] < lower_12h[i] and
                  bearish_aligned[i] > 0.5 and
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.30
                position = -1
        else:
            # Exit: price returns to midpoint of channel
            midpoint = (upper_12h[i] + lower_12h[i]) / 2
            if (position == 1 and close[i] < midpoint) or \
               (position == -1 and close[i] > midpoint):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0