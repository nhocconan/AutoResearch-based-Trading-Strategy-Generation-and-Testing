#!/usr/bin/env python3
"""
12h_KAMA_Trend_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend signals in both trending and ranging markets. Combined with volume confirmation and 1-day trend filter, it produces high-probability entries with low trade frequency suitable for 12h timeframe, working in both bull and bear markets.
"""

name = "12h_KAMA_Trend_Filter"
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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA (10, 2, 30)
    # ER = |net change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(close)
    for i in range(10, n):
        net_change = np.abs(close[i] - close[i-10])
        sum_change = np.sum(abs_change[i-9:i+1])
        if sum_change != 0:
            er[i] = net_change / sum_change
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d trend filter: EMA(34) on close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after KAMA warmup
        if position == 0:
            # LONG: Price crosses above KAMA, volume confirmation, price above 1d EMA34 (uptrend)
            if (close[i] > kama[i] and close[i-1] <= kama[i-1] and 
                volume_filter[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA, volume confirmation, price below 1d EMA34 (downtrend)
            elif (close[i] < kama[i] and close[i-1] >= kama[i-1] and 
                  volume_filter[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA OR volume drops
            if (close[i] < kama[i] and close[i-1] >= kama[i-1]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA OR volume drops
            if (close[i] > kama[i] and close[i-1] <= kama[i-1]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals