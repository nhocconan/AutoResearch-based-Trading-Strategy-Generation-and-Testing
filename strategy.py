#!/usr/bin/env python3
# 4h_200EMA_VWAP_Pullback
# Hypothesis: On 4h timeframe, enter long when price pulls back to 200EMA during uptrend (price > VWAP) and short when price rallies to 200EMA during downtrend (price < VWAP).
# Uses VWAP as trend filter and 200EMA as dynamic support/resistance. Works in bull markets (buy dips) and bear markets (sell rallies) by following institutional trend.
# Designed for 15-40 trades/year to avoid fee drag. Uses volume-weighted price for institutional-grade trend detection.

name = "4h_200EMA_VWAP_Pullback"
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

    # Calculate VWAP (volume-weighted average price)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den

    # Calculate 200 EMA on close
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required data is NaN
        if np.isnan(vwap[i]) or np.isnan(ema_200[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below VWAP
        above_vwap = close[i] > vwap[i]
        below_vwap = close[i] < vwap[i]

        if position == 0:
            # LONG: Price pulls back to 200EMA from above during uptrend (price > VWAP)
            if close[i] <= ema_200[i] and close[i-1] > ema_200[i-1] and above_vwap:
                signals[i] = 0.25
                position = 1
            # SHORT: Price rallies to 200EMA from below during downtrend (price < VWAP)
            elif close[i] >= ema_200[i] and close[i-1] < ema_200[i-1] and below_vwap:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 200EMA or trend changes
            if close[i] < ema_200[i] and close[i-1] >= ema_200[i-1] or not above_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 200EMA or trend changes
            if close[i] > ema_200[i] and close[i-1] <= ema_200[i-1] or not below_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals