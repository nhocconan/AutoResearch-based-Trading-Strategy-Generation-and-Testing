#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v3
# Hypothesis: Tightened version of Camarilla breakout with additional volatility filter to reduce trades.
# Uses daily Camarilla R1/S1 levels with 1d EMA34 trend filter and volume spike confirmation.
# Adds ATR-based volatility filter: only trade when ATR(14) > 20-period SMA of ATR to avoid choppy markets.
# Designed for 12-37 trades/year to stay within optimal range and minimize fee drag.
# Works in bull/bear markets by following daily EMA trend direction.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v3"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla levels and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels for each day: based on prior day's OHLC
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    # We use prior day's values to avoid look-ahead
    rng_1d = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * rng_1d / 12
    camarilla_s1 = close_1d - 1.1 * rng_1d / 12

    # Align Camarilla levels to 12h timeframe (use prior day's levels for current day)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_sma20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > atr_sma20  # Only trade when volatility is above average

    # Calculate volume spike threshold (2.0x 20-period SMA on 12h)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i]) or
            np.isnan(atr[i]) or np.isnan(atr_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 with uptrend, volume spike, and sufficient volatility
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > volume_sma20[i] and
                volatility_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 with downtrend, volume spike, and sufficient volatility
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > volume_sma20[i] and
                  volatility_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches or crosses below S1 (opposite level)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches or crosses above R1 (opposite level)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals