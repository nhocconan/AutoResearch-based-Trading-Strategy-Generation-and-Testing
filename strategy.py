#!/usr/bin/env python3
"""
6h_12h_1d_WeeklyBreakout_VolumeTrend_v2
Hypothesis: 6-hour breakouts from weekly high/low levels (based on prior week's close) with 12h trend filter and volume confirmation.
Targets 6h timeframe to reduce trade frequency (target: 15-35 trades/year) while using proven weekly structure.
Only takes long when price breaks above weekly high with volume spike and 12h uptrend, short when breaks below weekly low with volume spike and 12h downtrend.
Weekly high/low provides stronger support/resistance than daily pivots, working in both bull and bear markets via trend filter and volume confirmation.
"""

name = "6h_12h_1d_WeeklyBreakout_VolumeTrend_v2"
timeframe = "6h"
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
    
    # Volume spike: >2.0x 30-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d data for weekly high/low calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Weekly high/low from previous 5 trading days (prior week)
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
    
    # Align weekly high/low to 6h timeframe (wait for weekly bar to close)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1d, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1d, weekly_low)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(weekly_high_aligned[i]) or
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly high + volume spike + price above 12h EMA50
            if (close[i] > weekly_high_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly low + volume spike + price below 12h EMA50
            elif (close[i] < weekly_low_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters between weekly low and high OR closes below 12h EMA50
            if (close[i] > weekly_low_aligned[i] and close[i] < weekly_high_aligned[i]) or \
               close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters between weekly low and high OR closes above 12h EMA50
            if (close[i] > weekly_low_aligned[i] and close[i] < weekly_high_aligned[i]) or \
               close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals