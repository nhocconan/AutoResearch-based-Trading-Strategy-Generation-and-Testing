#!/usr/bin/env python3
# 1d_KAMA_With_1wTrend_Filter
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) on daily timeframe adapts to market noise,
# providing trend direction that is less whipsaw-prone in both bull and bear markets.
# Weekly trend filter (EMA34) ensures we only trade in the direction of the higher timeframe trend.
# Entry occurs when price crosses KAMA with volume confirmation, reducing false signals.
# Target: 10-25 trades/year on 1d timeframe.

name = "1d_KAMA_With_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_fast=2, er_slow=30):
    """Kaufman Adaptive Moving Average"""
    close_series = pd.Series(close)
    change = abs(close_series.diff(er_slow))
    volatility = close_series.diff().abs().rolling(window=er_slow, min_periods=er_slow).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for KAMA
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on daily timeframe
    kama_vals = kama(close, er_fast=2, er_slow=30)
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (30) + weekly EMA (34) + volume EMA (20)
    start_idx = max(30, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_vals[i]) or 
            np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA AND weekly trend is up (price > EMA34) AND volume
            if close[i] > kama_vals[i] and close[i-1] <= kama_vals[i-1] and close[i] > ema34_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA AND weekly trend is down (price < EMA34) AND volume
            elif close[i] < kama_vals[i] and close[i-1] >= kama_vals[i-1] and close[i] < ema34_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama_vals[i] and close[i-1] >= kama_vals[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama_vals[i] and close[i-1] <= kama_vals[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals