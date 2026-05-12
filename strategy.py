#!/usr/bin/env python3
# 4h_1D_Camarilla_R1_S1_Breakout_Trend_VolumeS_v3
# Hypothesis: Further refined version with even stricter entry conditions to reduce trade frequency and improve robustness. Uses 2x volume spike requirement and adds a minimum price change filter to avoid whipsaws. Maintains the core logic of trading breakouts from daily Camarilla R1/S1 levels in the direction of the daily trend, but with stronger filters to ensure only high-quality signals are taken. Targets 50-100 total trades over 4 years to minimize fee drag while maintaining edge in both bull and bear markets by following higher-timeframe trend.

name = "4h_1D_Camarilla_R1_S1_Breakout_Trend_VolumeS_v3"
timeframe = "4h"
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

    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels (R1, S1) from previous day
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12

    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate 4h volume SMA20 for volume confirmation (with spike filter)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0  # Require 2x average volume (stricter)

    # Calculate price change filter to avoid whipsaws
    price_change = np.abs(close - np.roll(close, 1))
    price_change[0] = 0
    price_change_sma = pd.Series(price_change).rolling(window=10, min_periods=10).mean().values
    price_change_threshold = price_change_sma * 1.5  # Require 1.5x average price change

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i]) or
            np.isnan(price_change_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R1 in 1d uptrend with volume spike and price change confirmation
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > volume_spike_threshold[i] and
                price_change[i] > price_change_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S1 in 1d downtrend with volume spike and price change confirmation
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > volume_spike_threshold[i] and
                  price_change[i] > price_change_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1d EMA34 (trend change)
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 1d EMA34 (trend change)
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals