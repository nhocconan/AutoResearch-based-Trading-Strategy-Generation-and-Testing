#!/usr/bin/env python3
"""
4H_DONCHIAN_BREAKOUT_VOLUME_TREND
Hypothesis: Breakout above/below Donchian(20) channel with volume confirmation and 4h EMA50 trend filter.
This strategy captures momentum moves in both bull and bear markets by combining price channel breakouts
with volume confirmation and trend alignment. The Donchian channel provides clear breakout levels,
volume confirms institutional interest, and EMA50 ensures trades are taken in the direction of the 4h trend.
Target: 25-40 trades/year per symbol.
"""

name = "4H_DONCHIAN_BREAKOUT_VOLUME_TREND"
timeframe = "4h"
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
    
    # Donchian channel (20-period high/low)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA50 for trend filter on 4h
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    # 1d trend filter (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        if np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend direction: above/below 1d EMA200
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # LONG: Break above Donchian upper band with volume confirmation and bullish trend
            if (high[i] > high_roll[i] and 
                volume_confirm[i] and 
                bullish_trend):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian lower band with volume confirmation and bearish trend
            elif (low[i] < low_roll[i] and 
                  volume_confirm[i] and 
                  bearish_trend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below Donchian middle or trend turns bearish
            donchian_mid = (high_roll[i] + low_roll[i]) / 2.0
            if (close[i] < donchian_mid or 
                not bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above Donchian middle or trend turns bullish
            donchian_mid = (high_roll[i] + low_roll[i]) / 2.0
            if (close[i] > donchian_mid or 
                not bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals