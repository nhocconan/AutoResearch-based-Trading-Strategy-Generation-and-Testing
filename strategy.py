#!/usr/bin/env python3
"""
12H_DONCHIAN_BREAKOUT_VOLUME_TREND
Hypothesis: Use 12h Donchian channel breakouts with volume confirmation and 1d trend filter (EMA200) to capture strong momentum moves. The 1d EMA200 filter ensures trades align with long-term trend, reducing whipsaws in both bull and bear markets. Volume confirmation filters out weak breakouts. Target: 15-30 trades/year.
"""
name = "12H_DONCHIAN_BREAKOUT_VOLUME_TREND"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    # Volume spike: current 12h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    # 12h Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=1).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=1).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if np.isnan(ema200_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend: above EMA200 = bullish, below = bearish
        bullish_trend = close[i] > ema200_aligned[i]
        bearish_trend = close[i] < ema200_aligned[i]
        
        if position == 0:
            # LONG: Break above 20-period high with volume spike and bullish trend
            if (high[i] > high_20[i-1] and 
                volume_spike[i] and 
                bullish_trend):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below 20-period low with volume spike and bearish trend
            elif (low[i] < low_20[i-1] and 
                  volume_spike[i] and 
                  bearish_trend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below 20-period low or trend turns bearish
            if (close[i] < low_20[i-1] or 
                not bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above 20-period high or trend turns bullish
            if (close[i] > high_20[i-1] or 
                not bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals