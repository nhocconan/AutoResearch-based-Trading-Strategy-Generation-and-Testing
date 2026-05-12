#!/usr/bin/env python3
"""
4h_ChaikinMoneyFlow_DonchianBreakout_TrendFilter
Hypothesis: Combine Donchian breakout with Chaikin Money Flow accumulation/distribution and daily trend filter for high-conviction trades. Works in bull/bear by requiring price to break structure with institutional flow (CMF) and higher timeframe trend alignment, reducing false signals in chop. Target: 20-30 trades/year.
"""

name = "4h_ChaikinMoneyFlow_DonchianBreakout_TrendFilter"
timeframe = "4h"
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

    # Get daily data for trend filter (call once before loop)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    close_daily = df_daily['close'].values
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)

    # Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Chaikin Money Flow (20-period)
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where(high == low, 0, mfm)  # avoid division by zero
    mfv = mfm * volume
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = mfv_sum / vol_sum

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        donch_high = highest_high[i]
        donch_low = lowest_low[i]
        cmf_val = cmf[i]
        ema50_val = ema50_daily_aligned[i]

        if np.isnan(donch_high) or np.isnan(donch_low) or np.isnan(cmf_val) or np.isnan(ema50_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Donchian high + CMF positive + daily uptrend
            if close[i] > donch_high and cmf_val > 0.05 and close[i] > ema50_val:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low + CMF negative + daily downtrend
            elif close[i] < donch_low and cmf_val < -0.05 and close[i] < ema50_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian low OR CMF turns negative
            if close[i] < donch_low or cmf_val < -0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian high OR CMF turns positive
            if close[i] > donch_high or cmf_val > 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals