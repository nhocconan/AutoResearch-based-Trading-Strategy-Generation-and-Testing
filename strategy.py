#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_Trend_Volume
Hypothesis: Weekly pivot levels (from 1w) act as strong support/resistance. 
Donchian breakouts (20-period) in the direction of weekly pivot bias with volume confirmation 
work in both bull and bear markets. Weekly pivot provides structural context, 
Donchian captures breakouts, volume confirms conviction. 
Target: 15-35 trades/year per symbol.
"""

name = "6h_WeeklyPivot_Donchian_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel: 20-period high/low
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot levels (from 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H+L+C)/3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pivot_point = typical_price.values
    # Support 1: 2*P - H
    support_1 = (2 * pivot_point) - df_1w['high'].values
    # Resistance 1: 2*P - L
    resistance_1 = (2 * pivot_point) - df_1w['low'].values
    
    # Weekly trend bias: price above/below pivot
    weekly_bullish = df_1w['close'].values > pivot_point
    weekly_bearish = df_1w['close'].values < pivot_point
    
    # Align weekly data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    support_1_aligned = align_htf_to_ltf(prices, df_1w, support_1)
    resistance_1_aligned = align_htf_to_ltf(prices, df_1w, resistance_1)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        dh = donch_high[i]
        dl = donch_low[i]
        pivot = pivot_aligned[i]
        sup1 = support_1_aligned[i]
        res1 = resistance_1_aligned[i]
        w_bull = weekly_bullish_aligned[i]
        w_bear = weekly_bearish_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: Donchian breakout above resistance with weekly bullish bias and volume
            if close[i] > dh and w_bull and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: Donchian breakdown below support with weekly bearish bias and volume
            elif close[i] < dl and w_bear and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to pivot or weekly bias turns bearish
            if close[i] <= pivot or not w_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to pivot or weekly bias turns bullish
            if close[i] >= pivot or not w_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals