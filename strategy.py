#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1dTrend_Volume
Hypothesis: On 12-hour timeframe, Donchian channel breakouts (20-period) with
daily trend alignment and volume confirmation capture significant moves while
minimizing whipsaws. Exits on reversion to the 20-period EMA or opposite
breakout. Targets 15-25 trades/year to avoid fee drag in both bull and bear markets.
"""

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 100-period EMA for daily trend
    ema100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Volume confirmation: current volume > 1.5x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Calculate Donchian channels (20-period) on 12h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above 20-period high with volume and uptrend
            if (high[i] > high_max[i-1] and 
                volume_filter[i] and 
                close[i] > ema100_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below 20-period low with volume and downtrend
            elif (low[i] < low_min[i-1] and 
                  volume_filter[i] and 
                  close[i] < ema100_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 20-period EMA or opposite breakout
            ema20 = pd.Series(close[:i+1]).ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
            if close[i] < ema20 or low[i] < low_min[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 20-period EMA or opposite breakout
            ema20 = pd.Series(close[:i+1]).ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
            if close[i] > ema20 or high[i] > high_max[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals