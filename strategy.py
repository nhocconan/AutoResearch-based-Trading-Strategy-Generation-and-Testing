#!/usr/bin/env python3
# 4h_Performance_Index_Trend
# Hypothesis: Go long when 4h price closes above 4h EMA50 and 1d RSI > 50 (bullish momentum), short when below EMA50 and 1d RSI < 50 (bearish momentum).
# Exit when price crosses back over EMA50. Uses daily RSI for trend filter to avoid whipsaws in chop.
# Works in bull (follows uptrend) and bear (follows downtrend). Low frequency due to EMA50 crossover + RSI filter.
# Trend filter from higher timeframe reduces false signals. Target: ~25-40 trades/year.

name = "4h_Performance_Index_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values

    # Get daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Daily RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 4h EMA50
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(ema50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > EMA50 and daily RSI > 50
            if close[i] > ema50[i] and rsi_1d_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < EMA50 and daily RSI < 50
            elif close[i] < ema50[i] and rsi_1d_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA50
            if close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA50
            if close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals