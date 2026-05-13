#!/usr/bin/env python3
# 6h_Choppiness_Index_MeanReversion_1dTrend_Filter
# Hypothesis: In choppy markets (CHOP > 61.8), price mean-reverts at Bollinger Bands (20,2). 
# In trending markets (CHOP < 38.2), follow 1d EMA34 trend. Combines regime detection with 
# mean-reversion and trend-following to work in both bull and bear markets. Uses 6h timeframe 
# to limit trade frequency and reduce fee drag.

name = "6h_Choppiness_Index_MeanReversion_1dTrend_Filter"
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

    # ATR for Choppy Index calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Choppy Index (14)
    sum_atr_14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((range_14 == 0) | np.isnan(range_14), 50, chop)

    # Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean()
    bb_std = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = (bb_mid + 2 * bb_std).values
    bb_lower = (bb_mid - 2 * bb_std).values

    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(chop[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Mean-reversion in choppy market (CHOP > 61.8)
            if chop[i] > 61.8:
                if close[i] <= bb_lower[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= bb_upper[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Trend following in trending market (CHOP < 38.2)
            elif chop[i] < 38.2:
                if close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Neutral zone: no action
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion exit in chop, or trend exhaustion
            if chop[i] > 61.8 and close[i] >= bb_mid.iloc[i]:
                signals[i] = 0.0
                position = 0
            elif chop[i] < 38.2 and close[i] <= ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion exit in chop, or trend exhaustion
            if chop[i] > 61.8 and close[i] <= bb_mid.iloc[i]:
                signals[i] = 0.0
                position = 0
            elif chop[i] < 38.2 and close[i] >= ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals