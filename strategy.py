#!/usr/bin/env python3
"""
4h_Bollinger_Bands_Reversion_With_Volume_Spice_and_Trend_Filter
Hypothesis: Mean reversion at Bollinger Bands (20,2) with volume spike (>1.5x 20-period average) and trend filter (1d EMA50) captures reversals in both bull and bear markets. Bollinger Bands provide dynamic support/resistance, volume confirms conviction, and 1d EMA50 ensures trades align with higher-timeframe trend. Designed for low trade frequency (<50/year) to minimize fee drag.
"""

name = "4h_Bollinger_Bands_Reversion_With_Volume_Spice_and_Trend_Filter"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Bollinger Bands (20, 2) on 4h
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean()
    dev = close_series.rolling(window=20, min_periods=20).std()
    upper_band = (basis + 2 * dev).values
    lower_band = (basis - 2 * dev).values

    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume spike: >1.5x 20-period average (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after Bollinger Bands warmup
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches lower Bollinger Band + 1d EMA50 uptrend + volume spike
            if (close[i] <= lower_band[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches upper Bollinger Band + 1d EMA50 downtrend + volume spike
            elif (close[i] >= upper_band[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above the basis (mean reversion complete)
            if close[i] >= basis.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below the basis (mean reversion complete)
            if close[i] <= basis.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals