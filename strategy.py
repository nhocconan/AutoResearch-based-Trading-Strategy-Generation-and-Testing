#!/usr/bin/env python3

# 6h_1D_Bollinger_Band_Width_Regime_Mean_Reversion
# Hypothesis: Mean reversion in low volatility regimes (BB Width < 30th percentile) with
# Bollinger Band touch entries on 6h, filtered by 1d trend (price above/below 200 EMA).
# Works in both bull and bear markets by capturing mean reversion during consolidation
# periods, which occur frequently across market regimes. Low frequency (~20-40 trades/year)
# to minimize fee drag.

name = "6h_1D_Bollinger_Band_Width_Regime_Mean_Reversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter and BB width regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)

    # Calculate 1d Bollinger Bands and BB Width for regime filter
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d

    # BB Width regime: low volatility when BB Width < 30th percentile
    # Use expanding window percentile to avoid look-ahead
    bb_width_percentile = pd.Series(bb_width_1d).expanding(min_periods=50).quantile(0.30).values
    low_vol_regime = bb_width_1d < bb_width_percentile
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))

    # Calculate 6h Bollinger Bands for entry signals
    sma_20_6h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20_6h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb_6h = sma_20_6h + 2 * std_20_6h
    lower_bb_6h = sma_20_6h - 2 * std_20_6h

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(low_vol_regime_aligned[i]) or
            np.isnan(upper_bb_6h[i]) or np.isnan(lower_bb_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Regime and trend filters
        in_low_vol = low_vol_regime_aligned[i] > 0.5  # boolean as float
        bullish_trend = close[i] > ema_200_1d_aligned[i]
        bearish_trend = close[i] < ema_200_1d_aligned[i]

        if position == 0:
            # LONG: Price touches lower BB in low volatility regime with bullish 1d trend
            if close[i] <= lower_bb_6h[i] and in_low_vol and bullish_trend:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches upper BB in low volatility regime with bearish 1d trend
            elif close[i] >= upper_bb_6h[i] and in_low_vol and bearish_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches middle BB or regime/trend changes
            if close[i] >= sma_20_6h[i] or not in_low_vol or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches middle BB or regime/trend changes
            if close[i] <= sma_20_6h[i] or not in_low_vol or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals