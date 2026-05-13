#!/usr/bin/env python3
# 6h_Chaikin_Oscillator_ZeroCross_Trend_Filter
# Hypothesis: Chaikin Oscillator (3,10) zero-cross signals combined with 1d trend filter (EMA34) capture momentum shifts with institutional confirmation.
# Long when Chaikin Osc crosses above zero and price > 1d EMA34 (uptrend).
# Short when Chaikin Osc crosses below zero and price < 1d EMA34 (downtrend).
# Works in bull markets (captures early momentum) and bear markets (identifies distribution/accumulation phases).
# Uses volume-weighted accumulation/distribution to filter noise. Target: 15-30 trades/year per symbol.

name = "6h_Chaikin_Oscillator_ZeroCross_Trend_Filter"
timeframe = "6h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Chaikin Oscillator: (3-period EMA of ADL) - (10-period EMA of ADL)
    # ADL = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    # Handle division by zero when high == low
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1, hl_range)  # avoid div by zero
    adl = ((close - low) - (high - close)) / hl_range * volume
    adl = np.cumsum(adl)  # cumulative ADL
    
    # Calculate 3-period and 10-period EMA of ADL
    ema3_adl = pd.Series(adl).ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10_adl = pd.Series(adl).ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin_osc = ema3_adl - ema10_adl
    
    # 1d trend: EMA34
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Align Chaikin Oscillator to 6h timeframe
    chaikin_osc_aligned = align_htf_to_ltf(prices, df_1d, chaikin_osc)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(chaikin_osc_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Chaikin Osc crosses above zero + 1d uptrend
            if chaikin_osc_aligned[i] > 0 and chaikin_osc_aligned[i-1] <= 0 and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Chaikin Osc crosses below zero + 1d downtrend
            elif chaikin_osc_aligned[i] < 0 and chaikin_osc_aligned[i-1] >= 0 and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Chaikin Osc crosses below zero or trend reversal
            if chaikin_osc_aligned[i] < 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Chaikin Osc crosses above zero or trend reversal
            if chaikin_osc_aligned[i] > 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals