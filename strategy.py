#!/usr/bin/env python3
"""
12h_1d_200ema_Touch_With_Volume_and_1wTrend
Hypothesis: Price touching the 200-day EMA on the daily chart acts as strong support/resistance.
A touch above with bullish weekly trend and volume confirmation signals long.
A touch below with bearish weekly trend and volume confirmation signals short.
This strategy avoids frequent whipsaws by requiring alignment with the weekly trend,
making it effective in both bull and bear markets. Targets 12-37 trades/year.
"""

name = "12h_1d_200ema_Touch_With_Volume_and_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for 200 EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 200 EMA on daily close
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Proximity threshold: 0.5% of price
    proximity = 0.005 * ema_200_1d
    # Define touch zones
    touch_above = (close_1d >= ema_200_1d) & (close_1d <= ema_200_1d + proximity)
    touch_below = (close_1d <= ema_200_1d) & (close_1d >= ema_200_1d - proximity)
    
    # Align touch zones to 12h
    touch_above_aligned = align_htf_to_ltf(prices, df_1d, touch_above)
    touch_below_aligned = align_htf_to_ltf(prices, df_1d, touch_below)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 50 EMA on weekly close for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend: weekly close above 50 EMA
    uptrend_1w = close_1w > ema_50_1w
    # Downtrend: weekly close below 50 EMA
    downtrend_1w = close_1w < ema_50_1w
    
    # Align weekly trend to 12h
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Get aligned values for current bar
        touch_abv = touch_above_aligned[i]
        touch_blw = touch_below_aligned[i]
        uptrend = uptrend_1w_aligned[i]
        downtrend = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price touches above 200 EMA, weekly uptrend, volume confirmation
            if touch_abv and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: price touches below 200 EMA, weekly downtrend, volume confirmation
            elif touch_blw and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches below 200 EMA or weekly trend turns down
            if touch_blw or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches above 200 EMA or weekly trend turns up
            if touch_abv or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals