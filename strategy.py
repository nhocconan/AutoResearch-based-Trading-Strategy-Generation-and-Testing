#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dEMA50_Trend_Volume
Hypothesis: Donchian channel breakouts on 12h combined with 1d EMA50 trend filter and volume confirmation capture strong trending moves while avoiding whipsaws. The 12h timeframe reduces trade frequency to minimize fee drag, and the 1d EMA50 ensures alignment with the dominant trend. Volume confirmation adds conviction to breakouts. Works in both bull and bear markets by following the trend direction as defined by the higher timeframe.
Target: 15-35 trades/year per symbol.
"""

name = "12h_Donchian20_Breakout_1dEMA50_Trend_Volume"
timeframe = "12h"
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
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume)
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above prior period's high
        breakout_down = close[i] < donchian_low[i-1]  # Break below prior period's low
        
        if position == 0:
            # LONG: upward breakout + volume confirmation + 1d uptrend
            if breakout_up and volume_confirm[i] and uptrend_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: downward breakout + volume confirmation + 1d downtrend
            elif breakout_down and volume_confirm[i] and downtrend_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price closes below Donchian low or trend reverses
            if close[i] < donchian_low[i] or not uptrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price closes above Donchian high or trend reverses
            if close[i] > donchian_high[i] or not downtrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals