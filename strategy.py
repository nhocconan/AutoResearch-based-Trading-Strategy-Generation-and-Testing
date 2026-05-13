#!/usr/bin/env python3
# 1h_4H_1D_RSI_Reversal_v1
# Hypothesis: Use 4h RSI for momentum direction and 1d RSI for overbought/oversold extremes.
# Enter long when 4h RSI > 50 (bullish momentum) and 1d RSI < 30 (oversold) with price near 1h VWAP.
# Enter short when 4h RSI < 50 (bearish momentum) and 1d RSI > 70 (overbought) with price near 1h VWAP.
# Exit on opposite 1d RSI extreme or momentum reversal.
# Designed for low-frequency, high-conviction trades in ranging and trending markets.
# Uses 1h only for entry timing via VWAP proximity, reducing false signals.

name = "1h_4H_1D_RSI_Reversal_v1"
timeframe = "1h"
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

    # Calculate 1h VWAP for entry timing
    vwap_num = (high + low + close) / 3 * volume
    vwap_den = volume
    vwap = np.nancumsum(vwap_num) / np.nancumsum(vwap_den)
    # For first bar, avoid division by zero
    vwap[0] = (high[0] + low[0] + close[0]) / 3

    # Get 4h data for momentum RSI
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # RSI(14) on 4h
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    # RSI > 50 = bullish momentum, < 50 = bearish

    # Get 1d data for extreme RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # RSI(14) on 1d
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    # RSI < 30 = oversold, > 70 = overbought

    # Align 4h and 1d indicators to 1h
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Price near VWAP (within 0.5%)
        price_near_vwap = abs(close[i] - vwap[i]) / vwap[i] < 0.005

        # Momentum and extreme conditions
        rsi_4h_bullish = rsi_4h_aligned[i] > 50
        rsi_4h_bearish = rsi_4h_aligned[i] < 50
        rsi_1d_oversold = rsi_1d_aligned[i] < 30
        rsi_1d_overbought = rsi_1d_aligned[i] > 70

        if position == 0:
            # LONG: 4h bullish momentum + 1d oversold + price near VWAP
            if rsi_4h_bullish and rsi_1d_oversold and price_near_vwap:
                signals[i] = 0.20
                position = 1
            # SHORT: 4h bearish momentum + 1d overbought + price near VWAP
            elif rsi_4h_bearish and rsi_1d_overbought and price_near_vwap:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 1d RSI > 70 (overbought) or 4h momentum turns bearish
            if rsi_1d_aligned[i] > 70 or not rsi_4h_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 1d RSI < 30 (oversold) or 4h momentum turns bullish
            if rsi_1d_aligned[i] < 30 or not rsi_4h_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals