#!/usr/bin/env python3
# 1d_WeeklyPivot_Breakout_Trend_v2
# Hypothesis: Breakouts of weekly pivot levels (R1/S1) with 1-week trend filter (price > 1w EMA34) and volume confirmation.
# Works in both bull and bear markets: weekly pivots provide key levels that hold across regimes, 1w EMA34 ensures
# alignment with longer-term trend, volume confirms breakout strength. Targets 10-25 trades/year on 1d timeframe.
# Uses 1d timeframe as required by experiment.

name = "1d_WeeklyPivot_Breakout_Trend_v2"
timeframe = "1d"
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
    
    # Get 1w data for weekly pivot points and trend filter (HTF as specified)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 1d timeframe
    r1_1d = align_htf_to_ltf(prices, df_1w, r1)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1)
    ema_34_1w_1d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike filter on 1d (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or 
            np.isnan(ema_34_1w_1d[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > weekly R1, above 1w EMA34 trend, volume spike
            if close[i] > r1_1d[i] and close[i] > ema_34_1w_1d[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < weekly S1, below 1w EMA34 trend, volume spike
            elif close[i] < s1_1d[i] and close[i] < ema_34_1w_1d[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below weekly R1 or below 1w EMA34
            if close[i] < r1_1d[i] or close[i] < ema_34_1w_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above weekly S1 or above 1w EMA34
            if close[i] > s1_1d[i] or close[i] > ema_34_1w_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals