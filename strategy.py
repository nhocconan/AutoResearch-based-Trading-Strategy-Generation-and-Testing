#!/usr/bin/env python3
# 12h_ChaikinMoneyFlow_Trend_Filter
# Hypothesis: On 12h timeframe, trade long when CMF > 0.15 and price > weekly EMA50; short when CMF < -0.15 and price < weekly EMA50.
# Chaikin Money Flow measures institutional accumulation/distribution; weekly EMA50 filters trend.
# Designed for low turnover: only trade when strong money flow aligns with weekly trend.
# Works in bull (strong inflows in uptrend) and bear (strong outflows in downtrend) markets.

name = "12h_ChaikinMoneyFlow_Trend_Filter"
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

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate weekly EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Calculate Chaikin Money Flow (CMF) on 12h data
    # CMF = ADL(21) / Volume(21)
    # ADL = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    # Avoid division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # prevent div by zero
    clv = ((close - low) - (high - close)) / hl_range
    adl = clv * volume
    # Use pandas for rolling sum with min_periods
    adl_sum = pd.Series(adl).rolling(window=21, min_periods=21).sum().values
    vol_sum = pd.Series(volume).rolling(window=21, min_periods=21).sum().values
    cmf = adl_sum / vol_sum
    cmf = np.where(vol_sum == 0, 0, cmf)  # handle zero volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(cmf[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: CMF > 0.15 and price > weekly EMA50
            if (cmf[i] > 0.15 and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: CMF < -0.15 and price < weekly EMA50
            elif (cmf[i] < -0.15 and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF < 0 or trend breaks
            if (cmf[i] < 0 or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF > 0 or trend breaks
            if (cmf[i] > 0 or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals