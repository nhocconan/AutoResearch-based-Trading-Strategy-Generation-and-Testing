#!/usr/bin/env python3
"""
4h_Wilson_Shock_Absorber
Hypothesis: In bear markets, price often rejects after sharp moves due to liquidity exhaustion. 
This strategy uses Wilson's Shock Absorber (price deviation from EMA normalized by ATR) 
combined with 1-week RSI extremes and volume divergence to catch mean-reversion bounces 
in overextended moves. Works in both bull (buy dips) and bear (sell rallies) by fading 
extreme deviations when higher-timeframe momentum is exhausted.
"""

name = "4h_Wilson_Shock_Absorber"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def wilson_shock_absorber(close, ema, atr, c=2.0):
    """Wilson's Shock Absorber: (price - EMA) / (c * ATR)"""
    return (close - ema) / (c * atr)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate 1w RSI(14) for momentum exhaustion
    delta = np.diff(df_1w['close'], prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w[0:14] = np.nan  # Not enough data

    # Align 1w RSI to 4h
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)

    # Calculate 4h EMA(34) and ATR(14)
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values

    # Wilson Shock Absorber
    wsa = wilson_shock_absorber(close, ema_34, atr_14, c=2.5)
    
    # Volume divergence: decreasing volume on price extremes
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_decreasing = volume < vol_ma

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(wsa[i]) or 
            np.isnan(volume_decreasing[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Extreme negative WSA (oversold) + 1w RSI < 30 (bearish exhaustion) + volume decreasing
            if (wsa[i] < -1.2 and 
                rsi_1w_aligned[i] < 30 and 
                volume_decreasing[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Extreme positive WSA (overbought) + 1w RSI > 70 (bullish exhaustion) + volume decreasing
            elif (wsa[i] > 1.2 and 
                  rsi_1w_aligned[i] > 70 and 
                  volume_decreasing[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: WSA returns to neutral or RSI shows recovery
            if wsa[i] > -0.3 or rsi_1w_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: WSA returns to neutral or RSI shows weakness
            if wsa[i] < 0.3 or rsi_1w_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals