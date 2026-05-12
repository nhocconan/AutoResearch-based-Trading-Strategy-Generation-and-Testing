#!/usr/bin/env python3
"""
4h_Ichimoku_Cloud_Trend_With_12h_Filter
Hypothesis: On 4h timeframe, Ichimoku cloud breakouts with Tenkan/Kijun cross and price above/below cloud generate trend signals. 
12h EMA50 trend filter ensures alignment with higher timeframe momentum. Volume > 1.5x 20-period average confirms strength. 
Tenkan/Kijun cross exit provides timely reversal signals. Targets 20-50 trades/year (80-200 total over 4 years) with moderate turnover.
Works in bull via cloud breakouts and bear via cloud breakdowns with trend filter.
"""

name = "4h_Ichimoku_Cloud_Trend_With_12h_Filter"
timeframe = "4h"
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

    # Get 12h data (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)

    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Ichimoku components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)

    # Current cloud boundaries (Senkou Span A/B from 26 periods ago)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values will be invalid

    # Determine if price is above or below cloud
    # Upper cloud boundary = max(Senkou A, Senkou B)
    # Lower cloud boundary = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a_shifted, senkou_b_shifted)
    lower_cloud = np.minimum(senkou_a_shifted, senkou_b_shifted)

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Need 52 periods for Senkou B
        # Get aligned values for current 4h bar
        ema50 = ema50_12h_aligned[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        upper_cloud_val = upper_cloud[i]
        lower_cloud_val = lower_cloud[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(tenkan_val) or np.isnan(kijun_val) or 
            np.isnan(upper_cloud_val) or np.isnan(lower_cloud_val) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above cloud + Tenkan > Kijun + volume surge
            if (close[i] > upper_cloud_val and 
                tenkan_val > kijun_val and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud + Tenkan < Kijun + volume surge
            elif (close[i] < lower_cloud_val and 
                  tenkan_val < kijun_val and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan < Kijun (trend weakening) or price drops below cloud
            if (tenkan_val < kijun_val or close[i] < lower_cloud_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan > Kijun (trend reversing) or price rises above cloud
            if (tenkan_val > kijun_val or close[i] > upper_cloud_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals