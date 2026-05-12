#!/usr/bin/env python3
# 6h_1W_1D_Telegraph_Signal_With_Volume_Filter
# Hypothesis: Combines weekly trend (price above/below weekly VWAP) with daily momentum (RSI(14) > 50 for long, < 50 for short) and volume confirmation on 6h.
# Weekly VWAP acts as institutional trend filter; daily RSI provides momentum entry; volume spike confirms conviction.
# Works in bull markets (price above weekly VWAP + RSI > 50 + volume) and bear markets (price below weekly VWAP + RSI < 50 + volume).
# Targets 15-35 trades/year with discrete sizing to minimize fee drag.

name = "6h_1W_1D_Telegraph_Signal_With_Volume_Filter"
timeframe = "6h"
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

    # Get weekly data for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    # Calculate VWAP on weekly data: typical price * volume, cumulative
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    vwap_numerator = (typical_price_1w * df_1w['volume'].values).cumsum()
    vwap_denominator = df_1w['volume'].values.cumsum()
    vwap_1w = vwap_numerator / vwap_denominator

    # Align weekly VWAP to 6h timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)

    # Get daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)

    # Calculate RSI(14) on daily close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))

    # Align daily RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    # Calculate 20-period volume average on 6h for volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)  # 50% above average

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above weekly VWAP, RSI > 50, volume spike
            if close[i] > vwap_1w_aligned[i] and rsi_1d_aligned[i] > 50 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below weekly VWAP, RSI < 50, volume spike
            elif close[i] < vwap_1w_aligned[i] and rsi_1d_aligned[i] < 50 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly VWAP OR RSI < 40
            if close[i] < vwap_1w_aligned[i] or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly VWAP OR RSI > 60
            if close[i] > vwap_1w_aligned[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals