#!/usr/bin/env python3
"""
12h_Donchian20_Volume_Trend_W1_Trend
Hypothesis: On 12h timeframe, price breaking above/below 20-bar Donchian channel
with weekly trend confirmation (price above/below weekly EMA50) and volume surge
captures strong moves in both bull and bear markets. Weekly EMA50 acts as trend
filter to avoid counter-trend trades. Volume confirmation filters false breakouts.
Designed for low trade frequency (15-25/year) with clear exit at opposite Donchian
band to manage risk and reduce whipsaw.
"""

name = "12h_Donchian20_Volume_Trend_W1_Trend"
timeframe = "12h"
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
    
    # Weekly EMA50 for trend filter
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_12h = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG ENTRY: Price breaks above upper Donchian + volume + weekly uptrend
            if high[i] > high_roll[i-1] and volume_confirm[i] and close[i] > weekly_ema50_12h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT ENTRY: Price breaks below lower Donchian + volume + weekly downtrend
            elif low[i] < low_roll[i-1] and volume_confirm[i] and close[i] < weekly_ema50_12h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # LONG EXIT: Price reaches lower Donchian band (mean reversion) or trend change
            if low[i] < low_roll[i] or close[i] < weekly_ema50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # SHORT EXIT: Price reaches upper Donchian band or trend change
            if high[i] > high_roll[i] or close[i] > weekly_ema50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals