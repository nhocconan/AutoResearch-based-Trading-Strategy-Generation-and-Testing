#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Chop_Filter_v2
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for trend direction on daily timeframe,
# combined with RSI for momentum confirmation and Choppiness Index for regime filtering.
# Long when KAMA slopes up, RSI > 50, and market is trending (CHOP < 38.2).
# Short when KAMA slopes down, RSI < 50, and market is trending (CHOP < 38.2).
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Designed to reduce false signals in ranging markets and capture trends in both bull and bear markets.
# Target: 15-25 trades/year per symbol to minimize fee drag.

name = "1d_KAMA_Trend_RSI_Chop_Filter_v2"
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
    volume = prices['volume'].values

    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) == 0 else np.convolve(np.abs(np.diff(close)), np.ones(er_length), 'same')
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    # Calculate Choppiness Index
    def choppiness_index(high, low, close, cp_length=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high - low)
        tr2 = np.abs(np.roll(high, 1) - close)
        tr3 = np.abs(np.roll(low, 1) - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=cp_length, min_periods=cp_length).sum().values
        max_high = pd.Series(high).rolling(window=cp_length, min_periods=cp_length).max().values
        min_low = pd.Series(low).rolling(window=cp_length, min_periods=cp_length).min().values
        cpi = 100 * np.log10(atr / (max_high - min_low)) / np.log10(cp_length)
        return np.where((max_high - min_low) != 0, cpi, 50)

    # Calculate RSI
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA21 for trend
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)

    # Calculate daily indicators
    kama_vals = kama(close, er_length=10, fast_sc=2, slow_sc=30)
    rsi_vals = rsi(close, 14)
    chop_vals = choppiness_index(high, low, close, 14)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_vals[i]) or 
            np.isnan(rsi_vals[i]) or 
            np.isnan(chop_vals[i]) or 
            np.isnan(ema21_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising, RSI > 50, trending market (CHOP < 38.2), above weekly EMA21
            if kama_vals[i] > kama_vals[i-1] and rsi_vals[i] > 50 and chop_vals[i] < 38.2 and close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, RSI < 50, trending market (CHOP < 38.2), below weekly EMA21
            elif kama_vals[i] < kama_vals[i-1] and rsi_vals[i] < 50 and chop_vals[i] < 38.2 and close[i] < ema21_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling or RSI < 50 or choppy market (CHOP > 61.8)
            if kama_vals[i] < kama_vals[i-1] or rsi_vals[i] < 50 or chop_vals[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising or RSI > 50 or choppy market (CHOP > 61.8)
            if kama_vals[i] > kama_vals[i-1] or rsi_vals[i] > 50 or chop_vals[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals