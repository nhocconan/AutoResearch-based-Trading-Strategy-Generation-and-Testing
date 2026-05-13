#!/usr/bin/env python3
# 1d_1W_Keltner_Channel_Retest_With_Trend_Filter
# Hypothesis: In Bitcoin and Ethereum, price respects the 20-period EMA-based Keltner Channel
# on the daily timeframe. During trends, price pulls back to the 20 EMA (middle band) and
# bounces in the direction of the 1-week trend. The 1-week EMA50 determines the trend filter.
# Entry: Price touches the 20 EMA (within 0.5%) + weekly trend alignment + volume confirmation.
# Exit: Price reaches the opposite Keltner band (2*ATR) or trend reverses.
# This strategy avoids whipsaws by requiring weekly trend alignment and volume spikes,
# targeting only high-probability retests in trending markets. Designed for low turnover
# (<20 trades/year) to minimize fee drag in choppy or bear markets like 2025.

name = "1d_1W_Keltner_Channel_Retest_With_Trend_Filter"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values

    # Calculate EMA50 on weekly close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate ATR(20) for Keltner Channel on daily data
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values

    # Calculate EMA20 for Keltner Channel middle band
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Calculate upper and lower Keltner bands: EMA20 ± 2*ATR
    keltner_upper = ema20 + 2.0 * atr
    keltner_lower = ema20 - 2.0 * atr

    # Volume spike: volume > 1.5 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or
            np.isnan(ema20[i]) or
            np.isnan(keltner_upper[i]) or
            np.isnan(keltner_lower[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend (price > weekly EMA50) + price at EMA20 (within 0.5%) + volume spike
            if close[i] > ema50_1w_aligned[i] and \
               abs(close[i] - ema20[i]) / ema20[i] < 0.005 and \
               volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (price < weekly EMA50) + price at EMA20 (within 0.5%) + volume spike
            elif close[i] < ema50_1w_aligned[i] and \
                 abs(close[i] - ema20[i]) / ema20[i] < 0.005 and \
                 volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches upper Keltner band or trend turns bearish
            if close[i] >= keltner_upper[i] * 0.999 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches lower Keltner band or trend turns bullish
            if close[i] <= keltner_lower[i] * 1.001 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals