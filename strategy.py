#!/usr/bin/env python3
"""
1d_Choppiness_Index_Mean_Reversion
Hypothesis: In ranging markets (high Choppiness Index), price reverts to the mean (50-period SMA). 
Enter long when price is below SMA and Choppiness > 61.8, short when above SMA and Choppiness > 61.8.
Use 1-week trend filter to avoid trading against the major trend. Works in both bull and bear markets by adapting to range conditions.
Target: 10-25 trades/year per symbol.
"""

name = "1d_Choppiness_Index_Mean_Reversion"
timeframe = "1d"
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
    
    # 50-period SMA (mean)
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Choppiness Index (14-period)
    def true_range(h, l, c):
        tr1 = h[1:] - l[1:]
        tr2 = np.abs(h[1:] - c[:-1])
        tr3 = np.abs(l[1:] - c[:-1])
        return np.concatenate([[0.0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = pd.Series(true_range(high, low, close)).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_max_min = max_high - min_low
    range_max_min = np.where(range_max_min == 0, 1e-10, range_max_min)
    
    chop = 100 * np.log10(atr * 14 / range_max_min) / np.log10(14)
    
    # 1-week trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Only trade in ranging markets (Choppiness > 61.8)
        if chop[i] <= 61.8:
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get values
        price = close[i]
        sma = sma_50[i]
        uptrend_htf = uptrend_1w_aligned[i]
        downtrend_htf = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: price below SMA in ranging market, but only if 1w trend is not strongly down
            if price < sma and not downtrend_htf:
                signals[i] = 0.25
                position = 1
            # SHORT: price above SMA in ranging market, but only if 1w trend is not strongly up
            elif price > sma and not uptrend_htf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses above SMA or chop drops (trend emerging)
            if price >= sma or chop[i] < 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses below SMA or chop drops (trend emerging)
            if price <= sma or chop[i] < 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals