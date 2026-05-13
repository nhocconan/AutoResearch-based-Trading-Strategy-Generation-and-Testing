#!/usr/bin/env python3
"""
12h_Pivot_Reversion_With_Volume_Filter
Hypothesis: In ranging markets, price tends to revert to daily pivot levels. 
When price reaches daily pivot support/resistance with volume confirmation, 
we expect mean reversion. Uses 12h timeframe to reduce trade frequency and 
filters with daily trend to avoid counter-trend trades in strong trends.
Works in both bull and bear markets by focusing on mean reversion in ranges
while avoiding trending environments.
"""

name = "12h_Pivot_Reversion_With_Volume_Filter"
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
    
    # Get daily data for pivot points and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points: P = (H+L+C)/3, S1 = 2P - H, R1 = 2P - L
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3.0
    support_1 = 2 * pivot - daily_high
    resistance_1 = 2 * pivot - daily_low
    
    # Align daily pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    support_1_aligned = align_htf_to_ltf(prices, df_1d, support_1)
    resistance_1_aligned = align_htf_to_ltf(prices, df_1d, resistance_1)
    
    # Daily trend filter: EMA50
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price at or below daily S1 with volume confirmation and above daily EMA50 (avoid strong downtrends)
            if (close[i] <= support_1_aligned[i] and 
                volume_confirm[i] and 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at or above daily R1 with volume confirmation and below daily EMA50 (avoid strong uptrends)
            elif (close[i] >= resistance_1_aligned[i] and 
                  volume_confirm[i] and 
                  close[i] < trend_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches daily pivot or daily EMA50
            if (close[i] >= pivot_aligned[i] or 
                close[i] <= trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches daily pivot or daily EMA50
            if (close[i] <= pivot_aligned[i] or 
                close[i] >= trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals