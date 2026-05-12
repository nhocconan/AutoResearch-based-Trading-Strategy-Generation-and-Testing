#!/usr/bin/env python3
# 6h_1W_Ichimoku_CloudBreakout_Trend
# Hypothesis: Use weekly Ichimoku cloud for trend direction and 6h price breakout above/below cloud for entry.
# Weekly cloud acts as dynamic support/resistance; breaks indicate strong momentum in trend direction.
# Works in bull markets (breaks above cloud in uptrend) and bear markets (breaks below cloud in downtrend).
# Volume confirmation reduces false breaks. Target: 15-35 trades/year.

name = "6h_1W_Ichimoku_CloudBreakout_Trend"
timeframe = "6h"
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

    # Get weekly data for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate Ichimoku components (9, 26, 52 periods)
    high_9 = pd.Series(df_1w['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1w['low']).rolling(window=9, min_periods=9).min().values
    high_26 = pd.Series(df_1w['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1w['low']).rolling(window=26, min_periods=26).min().values
    high_52 = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).min().values

    tenkan_sen = (high_9 + low_9) / 2
    kijun_sen = (high_26 + low_26) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = (high_52 + low_52) / 2

    # Align to 6h timeframe (Ichimoku values are plotted 26 periods ahead)
    # But we use current values for cloud (no look-ahead)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)

    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)

    # Volume confirmation: current volume > 1.5x average of last 24 periods (4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine trend: price above/below cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]

        if position == 0:
            # LONG: Price breaks above cloud with volume confirmation
            if price_above_cloud and close[i-1] <= cloud_top[i-1] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below cloud with volume confirmation
            elif price_below_cloud and close[i-1] >= cloud_bottom[i-1] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters cloud (below cloud top)
            if close[i] < cloud_top[i] and close[i-1] >= cloud_top[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters cloud (above cloud bottom)
            if close[i] > cloud_bottom[i] and close[i-1] <= cloud_bottom[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals