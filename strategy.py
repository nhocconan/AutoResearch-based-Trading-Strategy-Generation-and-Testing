#!/usr/bin/env python3
"""
6H_FIBONACCI_PULLBACK_1D_TREND_RSI
Hypothesis: On 6b, pullbacks to 61.8% Fibonacci retracement of the prior 12h swing in the direction of 1d EMA50 trend, with RSI(14) < 40 for long and > 60 for short. Uses natural retracement levels in trending markets, effective in both bull and bear regimes. Target: 20-30 trades/year.
"""
name = "6H_FIBONACCI_PULLBACK_1D_TREND_RSI"
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
    
    # 12h data for swing high/low
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Identify swing high and low over last 2 12h bars
    swing_high = np.maximum.reduce([high_12h[-2:], high_12h[-1:]]) if len(high_12h) >= 2 else high_12h[-1]
    swing_low = np.minimum.reduce([low_12h[-2:], low_12h[-1:]]) if len(low_12h) >= 2 else low_12h[-1]
    
    # For array compatibility, compute per 12h bar
    swing_high_arr = np.maximum.accumulate(high_12h)
    swing_low_arr = np.minimum.accumulate(low_12h)
    
    # 61.8% Fibonacci level
    fib_range = swing_high_arr - swing_low_arr
    fib_618 = swing_low_arr + 0.618 * fib_range
    
    # Align to 6h
    fib_618_aligned = align_htf_to_ltf(prices, df_12h, fib_618)
    
    # 1d trend: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # wait for RSI warmup
        if (np.isnan(fib_618_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price near 61.8% fib, above 1d EMA50, RSI < 40
            if (low[i] <= fib_618_aligned[i] * 1.005 and  # within 0.5% of fib level
                high[i] >= fib_618_aligned[i] * 0.995 and
                close[i] > ema50_1d_aligned[i] and
                rsi[i] < 40):
                signals[i] = 0.25
                position = 1
            # SHORT: price near 61.8% fib, below 1d EMA50, RSI > 60
            elif (low[i] <= fib_618_aligned[i] * 1.005 and
                  high[i] >= fib_618_aligned[i] * 0.995 and
                  close[i] < ema50_1d_aligned[i] and
                  rsi[i] > 60):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses above swing high or RSI > 60
            if (high[i] > swing_high_arr[-1] if len(swing_high_arr) > 0 else False or 
                rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses below swing low or RSI < 40
            if (low[i] < swing_low_arr[-1] if len(swing_low_arr) > 0 else False or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals