#!/usr/bin/env python3
# 4h_Ichimoku_Breakout_VolumeTrend
# Hypothesis: Ichimoku cloud breakout with volume confirmation and trend filter on 4h timeframe.
# Long: Price breaks above Kumo (cloud) + volume > 1.5x SMA20 + price > Senkou Span A
# Short: Price breaks below Kumo + volume > 1.5x SMA20 + price < Senkou Span B
# Exit: Price re-enters the Kumo (cloud)
# Uses daily trend filter to avoid counter-trend trades in strong trends.

name = "4h_Ichimoku_Breakout_VolumeTrend"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Ichimoku calculations (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    tenkan_sen = (high_series.rolling(window=9, min_periods=9).max() + 
                  low_series.rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (high_series.rolling(window=26, min_periods=26).max() + 
                 low_series.rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((high_series.rolling(window=52, min_periods=52).max() + 
                 low_series.rolling(window=52, min_periods=52).min()) / 2).shift(26)
    # Kumo (Cloud): between Senkou Span A and Senkou Span B

    # Daily trend filter: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Ichimoku calculations are valid
        # Get aligned values for current 4h bar
        ema50_aligned = ema50_1d_aligned[i]
        vol_threshold_val = volume_threshold[i]

        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen.iloc[i]) or np.isnan(kijun_sen.iloc[i]) or 
            np.isnan(senkou_a.iloc[i]) or np.isnan(senkou_b.iloc[i]) or
            np.isnan(ema50_aligned) or np.isnan(vol_threshold_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Current Ichimoku values
        tenkan = tenkan_sen.iloc[i]
        kijun = kijun_sen.iloc[i]
        senkou_a_val = senkou_a.iloc[i]
        senkou_b_val = senkou_b.iloc[i]

        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)

        if position == 0:
            # LONG: Price breaks above cloud + volume spike + bullish alignment
            if (close[i] > cloud_top and
                volume[i] > vol_threshold_val and
                close[i] > ema50_aligned and
                tenkan > kijun):  # Additional bullish confirmation
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below cloud + volume spike + bearish alignment
            elif (close[i] < cloud_bottom and
                  volume[i] > vol_threshold_val and
                  close[i] < ema50_aligned and
                  tenkan < kijun):  # Additional bearish confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters the cloud (falls below cloud top)
            if close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters the cloud (rises above cloud bottom)
            if close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals