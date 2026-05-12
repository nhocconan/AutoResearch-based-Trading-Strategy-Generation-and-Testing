#!/usr/bin/env python3
"""
6h_Liquidity_Zone_Reversal
Hypothesis: Price often reverses at prior 12h liquidity zones (equal highs/lows) with confirmation from 1d trend and volume spike. Works in bull/bear by fading extremes and using trend filter to avoid counter-trend traps. Target: 15-25 trades/year.
"""

name = "6h_Liquidity_Zone_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for liquidity zones (equal highs/lows)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Identify equal highs/lows (liquidity zones) in 12h data
    # Equal high: current high == previous high (within 0.1% tolerance)
    # Equal low: current low == previous low (within 0.1% tolerance)
    eq_high = np.zeros(len(high_12h), dtype=bool)
    eq_low = np.zeros(len(low_12h), dtype=bool)
    for i in range(1, len(high_12h)):
        if abs(high_12h[i] - high_12h[i-1]) / high_12h[i-1] < 0.001:
            eq_high[i] = True
        if abs(low_12h[i] - low_12h[i-1]) / low_12h[i-1] < 0.001:
            eq_low[i] = True

    # Get liquidity zone levels
    liq_high = np.where(eq_high, high_12h, np.nan)
    liq_low = np.where(eq_low, low_12h, np.nan)

    # Forward fill to get the most recent liquidity level
    liq_high_series = pd.Series(liq_high)
    liq_low_series = pd.Series(liq_low)
    liq_high_ffill = liq_high_series.ffill().values
    liq_low_ffill = liq_low_series.ffill().values

    # Align to 6h timeframe
    liq_high_aligned = align_htf_to_ltf(prices, df_12h, liq_high_ffill)
    liq_low_aligned = align_htf_to_ltf(prices, df_12h, liq_low_ffill)

    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: current volume > 2.0x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        liq_high_val = liq_high_aligned[i]
        liq_low_val = liq_low_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(liq_high_val) or np.isnan(liq_low_val) or np.isnan(ema50_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at liquidity low (support) + uptrend + volume spike
            if (abs(close[i] - liq_low_val) / liq_low_val < 0.005 and  # within 0.5% of liquidity low
                close[i] > ema50_val and
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at liquidity high (resistance) + downtrend + volume spike
            elif (abs(close[i] - liq_high_val) / liq_high_val < 0.005 and  # within 0.5% of liquidity high
                  close[i] < ema50_val and
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches liquidity high or trend turns down
            if (abs(close[i] - liq_high_val) / liq_high_val < 0.005 or close[i] < ema50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches liquidity low or trend turns up
            if (abs(close[i] - liq_low_val) / liq_low_val < 0.005 or close[i] > ema50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals