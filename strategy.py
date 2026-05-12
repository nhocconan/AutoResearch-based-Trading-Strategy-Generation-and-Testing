#!/usr/bin/env python3
"""
6h_RSI_BullBearPower_Reversal_1dTrend
Hypothesis: Bull/Bear Power (Elder Ray) combined with RSI(14) and 1-day EMA50 trend filter.
Long when Bull Power > 0, RSI < 50, and price above 1d EMA50 (weak pullback in uptrend).
Short when Bear Power < 0, RSI > 50, and price below 1d EMA50 (weak bounce in downtrend).
Designed for 60-120 trades/year on 6h timeframe to work in both bull and bear markets
by buying weakness in uptrends and selling strength in downtrends.
"""

name = "6h_RSI_BullBearPower_Reversal_1dTrend"
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
    volume = prices['volume'].values

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate RSI(14) on 6h closes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # fill NaN with 50 for stability

    # Calculate Bull Power and Bear Power (Elder Ray)
    # Bull Power = High - EMA13(close)
    # Bear Power = Low - EMA13(close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        ema50_val = ema50_1d_aligned[i]
        rsi_val = rsi_values[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]

        if np.isnan(ema50_val) or np.isnan(rsi_val) or np.isnan(bull_val) or np.isnan(bear_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power positive (strength), RSI < 50 (not overbought), price above 1d EMA50 (uptrend)
            if bull_val > 0 and rsi_val < 50 and close[i] > ema50_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power negative (weakness), RSI > 50 (not oversold), price below 1d EMA50 (downtrend)
            elif bear_val < 0 and rsi_val > 50 and close[i] < ema50_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power turns negative OR RSI > 60 (overbought)
            if bear_val < 0 or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power turns positive OR RSI < 40 (oversold)
            if bull_val > 0 or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals