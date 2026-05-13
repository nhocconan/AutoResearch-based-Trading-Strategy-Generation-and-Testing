#!/usr/bin/env python3
# 4h_RSI_Stretch_Back_to_VWAP_With_Trend
# Hypothesis: Price reverts to VWAP after strong RSI moves, but only in direction of higher timeframe trend.
# Long when RSI(14) > 70 and price closes below VWAP with 1d uptrend (EMA50).
# Short when RSI(14) < 30 and price closes above VWAP with 1d downtrend.
# VWAP acts as dynamic support/resistance; RSI extremes signal exhaustion.
# Trend filter avoids counter-trend trades in strong moves.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_RSI_Stretch_Back_to_VWAP_With_Trend"
timeframe = "4h"
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

    # Calculate VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if np.isnan(rsi[i]) or np.isnan(vwap[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 70 (overbought) and price < VWAP and 1d uptrend
            if rsi[i] > 70 and close[i] < vwap[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 30 (oversold) and price > VWAP and 1d downtrend
            elif rsi[i] < 30 and close[i] > vwap[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 50 (neutral) or price > VWAP (mean reversion complete)
            if rsi[i] < 50 or close[i] > vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI > 50 (neutral) or price < VWAP (mean reversion complete)
            if rsi[i] > 50 or close[i] < vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals