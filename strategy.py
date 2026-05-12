#!/usr/bin/env python3
"""
6H_ELDER_RAY_BULL_POWER_BEAR_POWER_1D_TREND_FILTER
Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) 
with 1-day trend filter. In bull markets (1d EMA50 > EMA200), take Bull Power > 0 entries.
In bear markets (1d EMA50 < EMA200), take Bear Power < 0 entries. 
Uses 6EMA for responsiveness and 13EMA for Elder Ray core. 
Designed for ~15-30 trades/year on 6h to minimize fee drag in both bull and bear regimes.
"""
name = "6H_ELDER_RAY_BULL_POWER_BEAR_POWER_1D_TREND_FILTER"
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
    
    # Calculate EMAs for Elder Ray
    close_series = pd.Series(close)
    ema6 = close_series.ewm(span=6, adjust=False, min_periods=6).values
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Get 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).values
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Trend: 1 = bull (EMA50 > EMA200), -1 = bear (EMA50 < EMA200), 0 = no trend
    trend_1d = np.where(ema50_1d_aligned > ema200_1d_aligned, 1,
                        np.where(ema50_1d_aligned < ema200_1d_aligned, -1, 0))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after EMA13 warmup
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(trend_1d[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull market + Bull Power positive
            if trend_1d[i] == 1 and bull_power[i] > 0:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear market + Bear Power positive ( Bear Power = EMA13 - Low > 0 means bearish)
            elif trend_1d[i] == -1 and bear_power[i] > 0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear power becomes positive (momentum fading) OR trend turns bear
            if bear_power[i] > 0 or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull power becomes positive OR trend turns bull
            if bull_power[i] > 0 or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals