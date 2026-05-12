#!/usr/bin/env python3
# 12h_1w_MarketStructure_Breakout
# Hypothesis: Uses weekly market structure (higher highs/lows) and 1d trend filter on 12h timeframe.
# Long when price breaks above weekly higher high with 1d uptrend; short when breaks below weekly higher low with 1d downtrend.
# Volume confirmation (>2x 50-period average) ensures institutional participation.
# Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drift.
# Works in bull/bear markets by following weekly structure and 1d trend alignment.

name = "12h_1w_MarketStructure_Breakout"
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
    
    # Volume spike: >2x 50-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly data for market structure (higher highs/lows)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly higher highs and higher lows
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Higher High: current week high > previous week high
    weekly_higher_high = weekly_high > np.roll(weekly_high, 1)
    # Higher Low: current week low > previous week low
    weekly_higher_low = weekly_low > np.roll(weekly_low, 1)
    # Lower High: current week high < previous week high
    weekly_lower_high = weekly_high < np.roll(weekly_high, 1)
    # Lower Low: current week low < previous week low
    weekly_lower_low = weekly_low < np.roll(weekly_low, 1)
    
    # Handle first values
    weekly_higher_high[0] = False
    weekly_higher_low[0] = False
    weekly_lower_high[0] = False
    weekly_lower_low[0] = False
    
    # Daily trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly and daily indicators to 12h timeframe
    weekly_higher_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_higher_high)
    weekly_higher_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_higher_low)
    weekly_lower_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_lower_high)
    weekly_lower_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_lower_low)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(weekly_higher_high_aligned[i]) or
            np.isnan(weekly_higher_low_aligned[i]) or
            np.isnan(weekly_lower_high_aligned[i]) or
            np.isnan(weekly_lower_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Weekly higher high + volume spike + price above 1d EMA50
            if (weekly_higher_high_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly lower low + volume spike + price below 1d EMA50
            elif (weekly_lower_low_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly lower high OR price below 1d EMA50
            if (weekly_lower_high_aligned[i]) or \
               close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly higher low OR price above 1d EMA50
            if (weekly_higher_low_aligned[i]) or \
               close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals