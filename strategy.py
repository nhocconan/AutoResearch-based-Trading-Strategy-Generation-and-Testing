#!/usr/bin/env python3
# 1d_ChaikinMoneyFlow_1dTrend_WeeklyTrend
# Hypothesis: Chaikin Money Flow (CMF) detects accumulation/distribution. 
# Long when CMF > 0.1 + price > 200-day EMA + weekly EMA20 > weekly EMA50 (bullish weekly trend).
# Short when CMF < -0.1 + price < 200-day EMA + weekly EMA20 < weekly EMA50 (bearish weekly trend).
# Uses 200-day EMA as trend filter and weekly EMA crossover for higher timeframe trend confirmation.
# Target: 15-25 trades/year (~60-100 total over 4 years) to minimize fee drag.

name = "1d_ChaikinMoneyFlow_1dTrend_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Money Flow Multiplier and Money Flow Volume
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)  # avoid division by zero
    mfv = mfm * volume

    # Chaikin Money Flow (20-period)
    cmf = np.full_like(close, np.nan, dtype=float)
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0)

    # 200-day EMA for trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values

    # Get weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required value is NaN
        if (np.isnan(cmf[i]) or np.isnan(ema200[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: CMF > 0.1 (accumulation) + price > EMA200 + weekly EMA20 > EMA50
            if (cmf[i] > 0.1 and 
                close[i] > ema200[i] and
                ema20_1w_aligned[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: CMF < -0.1 (distribution) + price < EMA200 + weekly EMA20 < EMA50
            elif (cmf[i] < -0.1 and 
                  close[i] < ema200[i] and
                  ema20_1w_aligned[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF < 0 (distribution) or weekly trend turns bearish
            if (cmf[i] < 0 or ema20_1w_aligned[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF > 0 (accumulation) or weekly trend turns bullish
            if (cmf[i] > 0 or ema20_1w_aligned[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals