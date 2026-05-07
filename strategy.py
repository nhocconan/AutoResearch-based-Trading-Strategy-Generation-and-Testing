#!/usr/bin/env python3
# 12h_WeeklyPivot_Breakout_1dTrend
# Hypothesis: Uses weekly pivot points (calculated from prior week's high/low/close) to identify key support/resistance levels.
# Breakouts above weekly R1 or below S1 with volume confirmation and 1d trend filter (price > EMA50 for longs, < EMA50 for shorts).
# Weekly pivots provide structure that works in both bull and bear markets by identifying institutional levels.
# Target: 15-30 trades/year to stay within optimal frequency range and minimize fee drag.

name = "12h_WeeklyPivot_Breakout_1dTrend"
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
    
    # Get daily data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from 1d data (using prior week's H/L/C)
    # We'll resample 1d data to weekly manually since we can't use .resample()
    # Get weekly high, low, close from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Arrays to store weekly pivot levels
    weekly_high = np.full(len(close_1d), np.nan)
    weekly_low = np.full(len(close_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    # Calculate weekly values (assuming 7 days per week)
    for i in range(6, len(close_1d)):
        weekly_high[i] = np.max(high_1d[i-6:i+1])
        weekly_low[i] = np.min(low_1d[i-6:i+1])
        weekly_close[i] = close_1d[i]
    
    # Calculate pivot points: P = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly levels to 1d timeframe (already aligned since calculated from 1d)
    # Now align to 12h timeframe
    weekly_pivot_12h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_12h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_12h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period volume average on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(weekly_r1_12h[i]) or np.isnan(weekly_s1_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly R1 with volume and above 1d EMA50
            if close[i] > weekly_r1_12h[i] and close[i] > ema_50_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S1 with volume and below 1d EMA50
            elif close[i] < weekly_s1_12h[i] and close[i] < ema_50_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes below weekly pivot or below 1d EMA50
            if close[i] < weekly_pivot_12h[i] or close[i] < ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above weekly pivot or above 1d EMA50
            if close[i] > weekly_pivot_12h[i] or close[i] > ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals