#!/usr/bin/env python3
# 6h_ElderRay_Ray_Momentum
# Hypothesis: Elder Ray index (bull power = high - EMA13, bear power = EMA13 - low) captures bull/bear strength.
# Combine with 13-period EMA for trend and 14-period RSI for overbought/oversold filter.
# Go long when bull power > 0, price > EMA13, and RSI < 70 (avoid overextended longs).
# Go short when bear power > 0, price < EMA13, and RSI > 30 (avoid overextended shorts).
# Use 1d trend filter (EMA50) to align with higher timeframe direction.
# Targets 15-35 trades/year to minimize fee drag and work in both bull/bear markets via trend filter.

name = "6h_ElderRay_Ray_Momentum"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate EMA13 for Elder Ray and trend
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Elder Ray components
    bull_power = high - ema13
    bear_power = ema13 - low

    # RSI for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull power positive (bullish momentum) + price > EMA13 + RSI not overbought + 1d uptrend
            if (bull_power[i] > 0 and 
                close[i] > ema13[i] and
                rsi[i] < 70 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear power positive (bearish momentum) + price < EMA13 + RSI not oversold + 1d downtrend
            elif (bear_power[i] > 0 and 
                  close[i] < ema13[i] and
                  rsi[i] > 30 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear power becomes positive (momentum shifts) or RSI overbought
            if bear_power[i] > 0 or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull power becomes positive (momentum shifts) or RSI oversold
            if bull_power[i] > 0 or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals