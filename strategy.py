#!/usr/bin/env python3
# 4h_Chaikin_Oscillator_ZeroCross_Trend_Filter
# Hypothesis: Chaikin Oscillator (3,10) crossing zero with 1d EMA trend filter and volume confirmation
# captures momentum shifts with low trade frequency. Works in bull via uptrend longs and in bear via downtrend shorts.
# Target: 20-30 trades/year on 4h to minimize fee drag.

name = "4h_Chaikin_Oscillator_ZeroCross_Trend_Filter"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate Chaikin Oscillator: (3-period EMA of ADL) - (10-period EMA of ADL)
    # ADL = ADL_prev + ((close - low) - (high - close)) / (high - low) * volume
    # Avoid division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1, hl_range)  # replace zero with 1 to avoid div/0
    money_flow_multiplier = ((close - low) - (high - close)) / hl_range
    money_flow_volume = money_flow_multiplier * volume
    adl = np.cumsum(money_flow_volume)

    # EMA of ADL
    adl_series = pd.Series(adl)
    ema3_adl = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10_adl = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin_osc = ema3_adl - ema10_adl

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(chaikin_osc[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Chaikin Osc crosses above zero + price > 1d EMA34 + volume spike
            if (chaikin_osc[i] > 0 and chaikin_osc[i-1] <= 0 and
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = 0.25
                position = 1
            # SHORT: Chaikin Osc crosses below zero + price < 1d EMA34 + volume spike
            elif (chaikin_osc[i] < 0 and chaikin_osc[i-1] >= 0 and
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Chaikin Osc crosses below zero
            if chaikin_osc[i] < 0 and chaikin_osc[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Chaikin Osc crosses above zero
            if chaikin_osc[i] > 0 and chaikin_osc[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals