#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_Chop_Filter
# Hypothesis: KAMA direction on 4h combined with RSI and Choppiness regime filter.
# KAMA adapts to trend strength, reducing whipsaw in chop. RSI filters extreme readings.
# Choppiness index identifies trending vs ranging markets to apply appropriate logic.
# Designed for 20-50 trades/year per symbol, works in both bull and bear via regime adaptation.

name = "4h_KAMA_Direction_RSI_Chop_Filter"
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

    # Get daily data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate Choppiness Index (14-period)
    atr_1d = []
    tr_1d = np.maximum(np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])),
        np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    ))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop = np.where((max_high_14 - min_low_14) == 0, 50, chop)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)

    # Get 4h data for KAMA and RSI
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)

    # KAMA calculation (ER=10, fast=2, slow=30)
    close_4h = df_4h['close'].values
    change = np.abs(np.concatenate([[close_4h[0]], close_4h[:-1]]) - close_4h)
    volatility = np.sum(np.abs(np.diff(close_4h, prepend=close_4h[0])).reshape(-1, 10), axis=1)
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.full_like(close_4h, np.nan)
    kama[9] = close_4h[9]  # start after enough data
    for i in range(10, len(close_4h)):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close_4h[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)

    # RSI calculation (14-period)
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(13, np.nan), rsi[13:]])
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        chop_val = chop_aligned[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        close_val = close[i]

        # Regime filters based on Choppiness
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8

        if position == 0:
            # Enter based on regime
            if is_trending:
                # Trend following: KAMA direction
                if close_val > kama_val and rsi_val > 50 and rsi_val < 70:
                    signals[i] = 0.25
                    position = 1
                elif close_val < kama_val and rsi_val < 50 and rsi_val > 30:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Mean reversion: RSI extremes
                if rsi_val < 30 and close_val > kama_val:
                    signals[i] = 0.25
                    position = 1
                elif rsi_val > 70 and close_val < kama_val:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Neutral chop: no trade
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend reversal or RSI overbought
            if (close_val < kama_val) or (rsi_val > 70) or (not is_trending and is_ranging):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend reversal or RSI oversold
            if (close_val > kama_val) or (rsi_val < 30) or (not is_trending and is_ranging):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals