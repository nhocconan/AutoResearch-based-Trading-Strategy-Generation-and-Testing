#!/usr/bin/env python3
"""
4h_PriceChannel_Breakout_Trend_Volume
Hypothesis: On 4h timeframe, price breaking above/below Donchian(20) channels with 4h EMA50 trend filter and volume confirmation captures institutional breakouts while avoiding false signals. Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band). Volume surge confirms institutional participation, and EMA50 filter ensures alignment with intermediate-term trend. Target 25-40 trades/year to minimize fee drag.
"""
name = "4h_PriceChannel_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Donchian channel (20-period high/low)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if position == 0:
            # Long: price breaks above upper Donchian + EMA50 uptrend + volume surge
            if close[i] > high_roll[i] and close[i] > ema_50[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + EMA50 downtrend + volume surge
            elif close[i] < low_roll[i] and close[i] < ema_50[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below EMA50 (trend change) or opposite Donchian band
            if close[i] < ema_50[i] or close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        else:  # position == -1
            # Short exit: price closes above EMA50 (trend change) or opposite Donchian band
            if close[i] > ema_50[i] or close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals