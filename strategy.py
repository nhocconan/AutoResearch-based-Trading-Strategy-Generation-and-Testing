#!/usr/bin/env python3
# 4h_KAMA_Trend_RSI_Pullback
# Hypothesis: Price retraces to KAMA during strong trends, with RSI confirming momentum exhaustion.
# In uptrends: go long when price pulls back to KAMA with RSI < 40.
# In downtrends: go short when price pulls back to KAMA with RSI > 60.
# Uses 1d trend filter to align with higher timeframe momentum. Works in both bull and bear markets.
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "4h_KAMA_Trend_RSI_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d trend: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # KAMA on 4h
    def calculate_kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close, prepend=close[0])).rolling(window=length, min_periods=1).sum()
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    
    # RSI on 4h
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, length=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(kama[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at KAMA + 1d uptrend + RSI oversold
            if close[i] >= kama[i] * 0.998 and close[i] <= kama[i] * 1.002 and close[i] > ema34_1d_aligned[i] and rsi[i] < 40:
                signals[i] = 0.25
                position = 1
            # SHORT: Price at KAMA + 1d downtrend + RSI overbought
            elif close[i] >= kama[i] * 0.998 and close[i] <= kama[i] * 1.002 and close[i] < ema34_1d_aligned[i] and rsi[i] > 60:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above KAMA or trend reversal
            if close[i] > kama[i] * 1.005 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below KAMA or trend reversal
            if close[i] < kama[i] * 0.995 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals