#!/usr/bin/env python3

"""
1d_Ichimoku_Cloud_Breakout_WeeklyTrend
Hypothesis: On 1d timeframe, buy when price breaks above Kumo (cloud) with price > Senkou Span B and Tenkan > Kijun, confirming weekly uptrend via EMA50. Sell when price breaks below Kumo or trend reverses. Uses volume confirmation (>1.5x average) to filter false breakouts. Designed for 10-25 trades/year to minimize fee drift and work in both bull (trend following) and bear (counter-trend reversals at cloud edges).
"""

name = "1d_Ichimoku_Cloud_Breakout_WeeklyTrend"
timeframe = "1d"
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

    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate Ichimoku components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2

    # Kumo (cloud) boundaries: Senkou Span A and B
    # Upper cloud = max(senkou_a, senkou_b)
    # Lower cloud = min(senkou_a, senkou_b)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)

    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):
        # Skip if any required value is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above cloud (close > upper cloud) AND
            #       Tenkan > Kijun (bullish momentum) AND
            #       Price > Senkou Span B (strong bullish) AND
            #       Weekly uptrend (price > EMA50) AND
            #       Volume confirmation
            if (close[i] > upper_cloud[i] and 
                tenkan[i] > kijun[i] and
                close[i] > senkou_b[i] and
                close[i] > ema50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below cloud (close < lower cloud) AND
            #        Tenkan < Kijun (bearish momentum) AND
            #        Price < Senkou Span B (strong bearish) AND
            #        Weekly downtrend (price < EMA50) AND
            #        Volume confirmation
            elif (close[i] < lower_cloud[i] and 
                  tenkan[i] < kijun[i] and
                  close[i] < senkou_b[i] and
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below cloud OR momentum turns bearish OR weekly trend fails
            if (close[i] < lower_cloud[i] or 
                tenkan[i] < kijun[i] or 
                close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above cloud OR momentum turns bullish OR weekly trend fails
            if (close[i] > upper_cloud[i] or 
                tenkan[i] > kijun[i] or 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals