#!/usr/bin/env python3
"""
6h_KAMA_Trend_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility, providing a dynamic trend filter. 
In low volatility (range markets), KAMA stays close to price, reducing false signals. In high volatility (trends), 
KAMA follows with less lag. We go long when price crosses above KAMA with volume confirmation, short when price 
crosses below KAMA with volume confirmation. Uses 6h timeframe with 1d trend filter (price > 1d EMA50 for longs, 
price < 1d EMA50 for shorts). Designed to work in both bull and bear markets by adapting to volatility regimes.
"""

name = "6h_KAMA_Trend_Filter"
timeframe = "6h"
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
    
    # Calculate KAMA (30, 2, 30) - ER period 10, fast EMA 2, slow EMA 30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation: change over 10 periods / sum of absolute changes over 10 periods
    change_10 = np.abs(np.diff(close, n=10, prepend=close[:10]))
    abs_change_sum_10 = np.sum(np.abs(np.diff(close, n=1, prepend=close[0]))[:, None] * np.ones(10), axis=1) if False else None
    
    # Simplified: use pandas for ER calculation
    close_series = pd.Series(close)
    change = close_series.diff().abs()
    volatility = change.rolling(window=10, min_periods=10).sum()
    direction = (close_series - close_series.shift(10)).abs()
    er = direction / volatility
    er = er.fillna(0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        if position == 0:
            # LONG: price crosses above KAMA, volume confirmation, price above 1d EMA50 (uptrend)
            if (close[i] > kama[i] and close[i-1] <= kama[i-1] and 
                volume_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price crosses below KAMA, volume confirmation, price below 1d EMA50 (downtrend)
            elif (close[i] < kama[i] and close[i-1] >= kama[i-1] and 
                  volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA OR volume drops
            if (close[i] < kama[i] and close[i-1] >= kama[i-1]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA OR volume drops
            if (close[i] > kama[i] and close[i-1] <= kama[i-1]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals