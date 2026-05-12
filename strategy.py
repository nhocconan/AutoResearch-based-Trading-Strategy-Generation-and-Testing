#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_Filter
Hypothesis: On daily timeframe, KAMA trend direction combined with RSI mean reversion and 
choppiness regime filter (Chop > 61.8 = range) provides edge in both bull and bear markets.
KAMA adapts to market noise, reducing whipsaw. RSI < 40 in uptrend or > 60 in downtrend 
exploits mean reversion within the trend. Chop filter avoids trending markets where mean 
reversion fails. Targets 7-25 trades/year with low turnover to minimize fee drag.
"""

name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
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

    # Get weekly data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate KAMA on weekly close
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        dir = np.abs(np.subtract(close, np.roll(close, length)))
        vol = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 1 else np.nancumsum(change)
        # Avoid division by zero
        er = np.where(vol != 0, dir / vol, 0)
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out

    kama_1w = kama(close_1w, 10, 2, 30)
    kama_1w_prev = np.roll(kama_1w, 1)  # Previous week's KAMA for trend
    kama_1w_prev[0] = np.nan

    # KAMA trend: 1 if close > KAMA(prev), -1 if close < KAMA(prev)
    kama_trend = np.where(close_1w > kama_1w_prev, 1, -1)
    kama_trend[0] = 0  # First value invalid
    kama_trend_aligned = align_htf_to_ltf(prices, df_1w, kama_trend)

    # Calculate RSI(14) on daily close
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out

    rsi_14 = rsi(close, 14)
    rsi_14_aligned = align_htf_to_ltf(prices, prices, rsi_14)  # Self-align for same timeframe

    # Calculate Choppiness Index on weekly data
    def chop(high, low, close, length=14):
        atr = []
        for i in range(len(high)):
            if i == 0:
                tr = high[0] - low[0]
            else:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr.append(tr)
        atr_sum = pd.Series(atr).rolling(window=length, min_periods=length).sum().values
        highest_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        lowest_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        # Avoid division by zero
        denominator = highest_high - lowest_low
        chop_val = 100 * np.log10(atr_sum / denominator) / np.log10(length)
        chop_val = np.where(denominator != 0, chop_val, 50)  # Default to 50 if range=0
        return chop_val

    chop_1w = chop(high_1w, low_1w, close_1w, 14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Warmup for indicators
        # Get aligned values
        kama_trend = kama_trend_aligned[i]
        rsi_val = rsi_14_aligned[i]
        chop_val = chop_1w_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(kama_trend) or np.isnan(rsi_val) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Only trade in ranging markets (Chop > 61.8)
        if chop_val <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend (KAMA trend up) + RSI oversold (< 40)
            if kama_trend == 1 and rsi_val < 40:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (KAMA trend down) + RSI overbought (> 60)
            elif kama_trend == -1 and rsi_val > 60:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought (> 70) or trend change to down
            if rsi_val > 70 or kama_trend == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold (< 30) or trend change to up
            if rsi_val < 30 or kama_trend == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals