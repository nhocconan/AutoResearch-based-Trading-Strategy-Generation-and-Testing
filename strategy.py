#!/usr/bin/env python3
# 12h_MultiTF_Trend_Strength_Momentum
# Hypothesis: Combining weekly trend strength (ADX), daily momentum (ROC), and price action confirmation on 12h creates robust signals that work in both bull and bear markets by filtering for strong momentum while avoiding chop. Uses weekly ADX > 25 for trend strength, daily ROC > 0 for momentum, and 12h price above/below 20-period EMA for direction. Designed for low frequency (target: 15-30 trades/year) to minimize fee drag.

name = "12h_MultiTF_Trend_Strength_Momentum"
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

    # --- Weekly Trend Strength (ADX) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # True Range and Directional Movement
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    plus_dm = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0

    # Smoothing
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1w
    minus_di_1w = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w + 1e-10)
    adx_1w = pd.Series(dx_1w).ewm(span=14, adjust=False, min_periods=14).mean().values

    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)

    # --- Daily Momentum (ROC) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    roc_1d = (close_1d - np.roll(close_1d, 10)) / np.roll(close_1d, 10) * 100
    roc_1d[0:10] = 0
    roc_1d_aligned = align_htf_to_ltf(prices, df_1d, roc_1d)

    # --- 12h Price Trend (EMA20) ---
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(roc_1d_aligned[i]) or 
            np.isnan(ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Strong weekly trend (ADX>25) + positive daily momentum + price above EMA20
            if (adx_1w_aligned[i] > 25 and 
                roc_1d_aligned[i] > 0 and
                close[i] > ema20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Strong weekly trend (ADX>25) + negative daily momentum + price below EMA20
            elif (adx_1w_aligned[i] > 25 and 
                  roc_1d_aligned[i] < 0 and
                  close[i] < ema20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakening (ADX<20) or momentum fading
            if (adx_1w_aligned[i] < 20 or roc_1d_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakening or momentum fading
            if (adx_1w_aligned[i] < 20 or roc_1d_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals