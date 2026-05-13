#!/usr/bin/env python3
# 1D_ChaikinMoneyFlow_1wTrend_WeeklyTrend
# Hypothesis: Chaikin Money Flow (CMF) measures buying/selling pressure. 
# Weekly trend (1w EMA20) filters trades to avoid counter-trend moves.
# Entry when CMF crosses above/below zero with weekly trend alignment.
# Target: 8-15 trades/year (32-60 total over 4 years) to minimize fee drag.
# Works in bull via CMF>0 + uptrend, bear via CMF<0 + downtrend.

name = "1D_ChaikinMoneyFlow_1wTrend_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Chaikin Money Flow (CMF) calculation
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = 20-period sum of Money Flow Volume / 20-period sum of Volume
    hl_range = high - low
    # Avoid division by zero
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    mf_multiplier = ((close - low) - (high - close)) / hl_range
    mf_volume = mf_multiplier * volume
    
    # 20-period sums for CMF
    mf_volume_sum = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = mf_volume_sum / volume_sum
    # Replace inf/NaN from zero volume with 0
    cmf = np.where(np.isnan(cmf) | np.isinf(cmf), 0, cmf)

    # Get weekly data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):
        # Skip if any required value is NaN
        if np.isnan(cmf[i]) or np.isnan(ema20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: CMF > 0 (buying pressure) + weekly uptrend (price > EMA20)
            if cmf[i] > 0 and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: CMF < 0 (selling pressure) + weekly downtrend (price < EMA20)
            elif cmf[i] < 0 and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF turns negative or weekly trend breaks down
            if cmf[i] < 0 or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF turns positive or weekly trend breaks up
            if cmf[i] > 0 or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals