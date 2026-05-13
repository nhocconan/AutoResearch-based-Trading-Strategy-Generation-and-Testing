#!/usr/bin/env python3
"""
6h_Pivot_Bounce_Momentum
Hypothesis: Price tends to bounce off daily pivot points with momentum in 6h timeframe.
Long when price bounces above S1 with bullish momentum, short when rejected at R1 with bearish momentum.
Uses daily pivot levels as support/resistance and 6h RSI for momentum confirmation.
Target: 15-25 trades/year per symbol to avoid fee drag.
Works in both bull and bear markets as pivot levels adapt to price action.
"""

name = "6h_Pivot_Bounce_Momentum"
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
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get daily pivot data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate daily pivot points: P = (H+L+C)/3, S1 = 2P - H, R1 = 2P - L
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    s1 = 2 * pivot - df_1d['high']
    r1 = 2 * pivot - df_1d['low']
    
    # Align pivot levels to 6h timeframe (waits for daily close)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: price bounces above S1 with bullish momentum (RSI > 50 and rising)
            if close[i] > s1_aligned[i] and rsi[i] > 50 and rsi[i] > rsi[i-1]:
                signals[i] = 0.25
                position = 1
            # SHORT: price rejected at R1 with bearish momentum (RSI < 50 and falling)
            elif close[i] < r1_aligned[i] and rsi[i] < 50 and rsi[i] < rsi[i-1]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below S1 or momentum turns bearish
            if close[i] < s1_aligned[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above R1 or momentum turns bullish
            if close[i] > r1_aligned[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals