#!/usr/bin/env python3
"""
6h_ChaikinMoneyFlow_1dTrend_Reversal
Hypothesis: On 6h timeframe, Chaikin Money Flow (CMF) > 0.25 with 1d EMA50 uptrend 
signals accumulation (long); CMF < -0.25 with 1d EMA50 downtrend signals distribution (short).
Uses 60-period CMF to smooth noise. Only trades when price is near 6h VWAP (within 1%) 
to avoid chasing extended moves. Targets 15-35 trades/year (60-140 total over 4 years).
Works in bull via trend-aligned accumulation and in bear via distribution at resistance.
"""

name = "6h_ChaikinMoneyFlow_1dTrend_Reversal"
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

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 60-period Chaikin Money Flow
    # CMF = sum((close - low - (high - close)) / (high - low) * volume) / sum(volume)
    # Avoid division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    mf_multiplier = ((close - low) - (high - close)) / hl_range
    mf_volume = mf_multiplier * volume
    # Sum over 60 periods
    mf_volume_sum = pd.Series(mf_volume).rolling(window=60, min_periods=60).sum().values
    volume_sum = pd.Series(volume).rolling(window=60, min_periods=60).sum().values
    cmf = mf_volume_sum / volume_sum

    # Calculate 6h VWAP (typical price * volume) / volume
    typical_price = (high + low + close) / 3
    vwap_num = pd.Series(typical_price * volume).rolling(window=60, min_periods=60).sum().values
    vwap_den = pd.Series(volume).rolling(window=60, min_periods=60).sum().values
    vwap = vwap_num / vwap_den
    # Price deviation from VWAP (%)
    vwap_dev = np.abs((close - vwap) / vwap)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Get aligned values for current 6h bar
        ema50 = ema50_1d_aligned[i]
        cmf_val = cmf[i]
        vwap_dev_val = vwap_dev[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(cmf_val) or np.isnan(vwap_dev_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: CMF > 0.25 (accumulation) + 1d EMA50 uptrend + price near VWAP
            if (cmf_val > 0.25 and 
                close[i] > ema50 and 
                vwap_dev_val < 0.01):  # within 1% of VWAP
                signals[i] = 0.25
                position = 1
            # SHORT: CMF < -0.25 (distribution) + 1d EMA50 downtrend + price near VWAP
            elif (cmf_val < -0.25 and 
                  close[i] < ema50 and 
                  vwap_dev_val < 0.01):  # within 1% of VWAP
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF turns negative or price breaks below VWAP
            if (cmf_val < 0 or close[i] < vwap):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF turns positive or price breaks above VWAP
            if (cmf_val > 0 or close[i] > vwap):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals