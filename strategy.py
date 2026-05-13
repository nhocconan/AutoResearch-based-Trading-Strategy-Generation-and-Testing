#!/usr/bin/env python3
# 1d_KAMA_Trend_Filter_RSI
# Hypothesis: KAMA on 1d identifies trend direction, RSI on 1d provides overbought/oversold signals for mean reversion entries, and 1w trend filter ensures alignment with higher timeframe momentum.
# Long when: 1d price > KAMA (uptrend), RSI < 30 (oversold), and 1w close > 1w EMA50 (bullish higher timeframe).
# Short when: 1d price < KAMA (downtrend), RSI > 70 (overbought), and 1w close < 1w EMA50 (bearish higher timeframe).
# Exit when trend reverses (price crosses KAMA) or RSI reaches neutral (50).
# Uses weekly timeframe for trend filter to reduce whipsaw and focus on high-probability mean-reversion trades within the dominant trend.
# Target: 15-25 trades/year per symbol to minimize fee drag.

name = "1d_KAMA_Trend_Filter_RSI"
timeframe = "1d"
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

    # Get 1d data for KAMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    close_1d = df_1d['close'].values
    direction = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # ER scaled to smoothing constants for fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price > KAMA (uptrend), RSI < 30 (oversold), 1w close > EMA50 (bullish)
            if close[i] > kama_aligned[i] and rsi_aligned[i] < 30 and close_1w[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < KAMA (downtrend), RSI > 70 (overbought), 1w close < EMA50 (bearish)
            elif close[i] < kama_aligned[i] and rsi_aligned[i] > 70 and close_1w[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: trend reversal (price < KAMA) or RSI reaches neutral (50)
            if close[i] < kama_aligned[i] or rsi_aligned[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: trend reversal (price > KAMA) or RSI reaches neutral (50)
            if close[i] > kama_aligned[i] or rsi_aligned[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals