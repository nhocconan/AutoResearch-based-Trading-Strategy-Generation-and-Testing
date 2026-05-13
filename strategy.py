#!/usr/bin/env python3
# 12h_KAMA_Direction_RSI_Pullback_TrendFilter
# Hypothesis: Use 1d KAMA trend direction with RSI pullback entries on 12h timeframe.
# In uptrend (price > KAMA), buy RSI pullbacks below 40; in downtrend (price < KAMA), sell RSI bounces above 60.
# Adds 1w trend filter to avoid counter-trend trades in strong markets. Works in both bull and bear by
# following the higher timeframe trend while capturing mean-reversion entries.
# Target: 15-30 trades/year to minimize fee drag.

name = "12h_KAMA_Direction_RSI_Pullback_TrendFilter"
timeframe = "12h"
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

    # Get 1d data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    close_1d = df_1d['close']
    change = abs(close_1d.diff())
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros(len(close_1d))
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[i-1])
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close']
    ema200_1w = close_1w.ewm(span=200, min_periods=200).mean().values
    
    # RSI on 12h price
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values

    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        kama_val = kama_aligned[i]
        ema200_val = ema200_1w_aligned[i]
        rsi_val = rsi[i]

        if position == 0:
            # LONG: price > KAMA (uptrend) AND RSI < 40 (pullback) AND price > 200w EMA (bull regime)
            if close[i] > kama_val and rsi_val < 40 and close[i] > ema200_val:
                signals[i] = 0.25
                position = 1
            # SHORT: price < KAMA (downtrend) AND RSI > 60 (bounce) AND price < 200w EMA (bear regime)
            elif close[i] < kama_val and rsi_val > 60 and close[i] < ema200_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < KAMA (trend change) OR RSI > 70 (overbought)
            if close[i] < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > KAMA (trend change) OR RSI < 30 (oversold)
            if close[i] > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals