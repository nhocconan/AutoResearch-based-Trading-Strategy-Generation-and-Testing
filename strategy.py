#!/usr/bin/env python3
"""
1d_Fibonacci_Retracement_Trend_Filter
Hypothesis: In both bull and bear markets, price retraces to key Fibonacci levels (38.2%, 50%, 61.8%) during pullbacks within a strong weekly trend. Enter long at 61.8% retracement in weekly uptrend, short at 38.2% retracement in weekly downtrend. Use volume confirmation to avoid false signals. Target: 15-25 trades/year per symbol.
"""

name = "1d_Fibonacci_Retracement_Trend_Filter"
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
    
    # Weekly trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # 20-day high/low for Fibonacci calculation
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Fibonacci levels: 38.2%, 50%, 61.8%
    diff = high_20 - low_20
    fib_382 = high_20 - 0.382 * diff
    fib_500 = high_20 - 0.500 * diff
    fib_618 = high_20 - 0.618 * diff
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        fib382 = fib_382[i]
        fib500 = fib_500[i]
        fib618 = fib_618[i]
        uptrend = uptrend_1w_aligned[i]
        downtrend = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price at 61.8% retracement in weekly uptrend with volume confirmation
            if close[i] <= fib618 and close[i] >= fib500 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: price at 38.2% retracement in weekly downtrend with volume confirmation
            elif close[i] >= fib382 and close[i] <= fib500 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches 50% retracement or weekly trend turns down
            if close[i] >= fib500 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches 50% retracement or weekly trend turns up
            if close[i] <= fib500 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals