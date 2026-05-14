#!/usr/bin/env python3
# 6h_Altman_Zscore_Reversal_1dTrend_Filter
# Hypothesis: In 6h timeframe, extreme deviations from the 1-day VWAP (Z-score > 2.0 or < -2.0)
# signal mean-reversion opportunities, but only when aligned with the 1-day trend (price vs EMA50).
# This captures reversals in overextended moves while avoiding counter-trend trades in strong trends.
# Designed for low-frequency, high-probability setups in both bull and bear markets.

name = "6h_Altman_Zscore_Reversal_1dTrend_Filter"
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

    # Calculate VWAP for each 6h bar
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = vwap_num / vwap_den

    # Calculate rolling standard deviation of price-VWAP deviation
    deviation = typical_price - vwap
    vol = pd.Series(deviation).rolling(window=24, min_periods=24).std().values  # 24*6h = 6 days
    zscore = deviation / vol

    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # Calculate 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(zscore[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Extreme negative deviation (oversold) + uptrend bias
            if zscore[i] < -2.0 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Extreme positive deviation (overbought) + downtrend bias
            elif zscore[i] > 2.0 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Deviation normalizes or trend turns bearish
            if zscore[i] > -0.5 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Deviation normalizes or trend turns bullish
            if zscore[i] < 0.5 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals