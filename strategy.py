#!/usr/bin/env python3
# 4H_KAMA_DIRECTION_RSI_FILTER
# Hypothesis: Kaufman's Adaptive Moving Average (KAMA) captures adaptive trend direction.
# In 1d uptrend (price > EMA34), go long when KAMA(14) is rising; in downtrend, go short when KAMA(14) is falling.
# RSI(14) filter avoids overbought/oversold extremes (RSI < 70 for long, RSI > 30 for short).
# Works in both bull and bear markets: trend filter prevents counter-trend trades, KAMA captures momentum within trend.
# Target: 20-30 trades/year on 4h timeframe.

name = "4H_KAMA_DIRECTION_RSI_FILTER"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Daily data for KAMA and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # KAMA calculation (ER = Efficiency Ratio, SC = Smoothing Constant)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) + 2 / (30 + 1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # RSI(14) for overbought/oversold filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Rising/falling KAMA detection (1-bar change)
    kama_rising = kama_aligned > np.roll(kama_aligned, 1)
    kama_falling = kama_aligned < np.roll(kama_aligned, 1)
    # Handle first bar
    kama_rising[0] = False
    kama_falling[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + KAMA rising + RSI not overbought
            if (close[i] > ema34_aligned[i] and 
                kama_rising[i] and 
                rsi_aligned[i] < 70):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + KAMA falling + RSI not oversold
            elif (close[i] < ema34_aligned[i] and 
                  kama_falling[i] and 
                  rsi_aligned[i] > 30):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or KAMA falling
            if (close[i] <= ema34_aligned[i] or 
                not kama_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or KAMA rising
            if (close[i] >= ema34_aligned[i] or 
                not kama_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals