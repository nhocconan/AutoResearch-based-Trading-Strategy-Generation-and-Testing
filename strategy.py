#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_ChopFilter
# Hypothesis: On daily timeframe, use KAMA to determine trend direction, RSI(14) for overbought/oversold conditions, and Choppiness Index to filter ranging markets. Enter long when KAMA trending up, RSI < 40, and CHOP > 61.8 (ranging). Enter short when KAMA trending down, RSI > 60, and CHOP > 61.8. Exit when RSI crosses 50 or CHOP < 38.2 (trending). Designed to work in both bull and bear markets by capturing mean reversion in ranging conditions while avoiding strong trends. Targets 10-20 trades per year to minimize fee drag.

name = "1d_KAMA_Direction_RSI_ChopFilter"
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

    # Get weekly data for trend context and chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values

    # Calculate KAMA (Kaufman Adaptive Moving Average) on weekly close
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing Constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals

    kama_1w = kama(close_1w, length=10, fast=2, slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)

    # Calculate RSI(14) on daily close
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals

    rsi_1d = rsi(close, length=14)

    # Calculate Choppiness Index on weekly data
    def choppiness_index(high, low, close, length=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of TR over period
        atr_sum = pd.Series(tr).rolling(window=length, min_periods=length).sum().values
        
        # Highest high and lowest low over period
        highest_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        lowest_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        
        # Chop = 100 * log10(ATRsum / (HH - LL)) / log10(length)
        hh_ll = highest_high - lowest_low
        chop = np.where(hh_ll > 0, 100 * np.log10(atr_sum / hh_ll) / np.log10(length), 50)
        return chop

    chop_1w = choppiness_index(high_1w, low_1w, close_1w, length=14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_1d[i]) or 
            np.isnan(chop_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA trending up (price > KAMA), RSI oversold (<40), Chop > 61.8 (ranging)
            if (close[i] > kama_1w_aligned[i] and 
                rsi_1d[i] < 40 and 
                chop_1w_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA trending down (price < KAMA), RSI overbought (>60), Chop > 61.8 (ranging)
            elif (close[i] < kama_1w_aligned[i] and 
                  rsi_1d[i] > 60 and 
                  chop_1w_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses above 50 OR Chop < 38.2 (trending)
            if rsi_1d[i] >= 50 or chop_1w_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses below 50 OR Chop < 38.2 (trending)
            if rsi_1d[i] <= 50 or chop_1w_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals