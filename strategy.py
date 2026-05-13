#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_12hVolume
# Hypothesis: 6h Ichimoku Cloud with 12h volume confirmation and 1d trend filter.
# Uses Tenkan/Kijun cross for entry, price above/below cloud for trend direction,
# and 12h volume spike for confirmation. Avoids range-bound markets by requiring
# price to be outside cloud. Works in bull/bear by only taking trades in
# direction of 1d EMA50 trend.

name = "6h_Ichimoku_Cloud_Trend_12hVolume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h and 1d data for HTF filters
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')

    # Ichimoku components (9, 26, 52) on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (high9 + low9) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (high26 + low26) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((high52 + low52) / 2)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # 12h volume spike: current volume > 2x 20-period average
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (2.0 * vol_ma_12h_aligned)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after sufficient warmup for Ichimoku
        # Skip if any required value is NaN
        if (np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])

        if position == 0:
            # LONG: Tenkan crosses above Kijun, price above cloud, uptrend, volume spike
            if (tenkan[i] > kijun[i] and 
                tenkan[i-1] <= kijun[i-1] and  # crossover confirmation
                close[i] > cloud_top and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan crosses below Kijun, price below cloud, downtrend, volume spike
            elif (tenkan[i] < kijun[i] and 
                  tenkan[i-1] >= kijun[i-1] and  # crossover confirmation
                  close[i] < cloud_bottom and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below cloud or trend turns down
            if close[i] < cloud_bottom or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above cloud or trend turns up
            if close[i] > cloud_top or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals