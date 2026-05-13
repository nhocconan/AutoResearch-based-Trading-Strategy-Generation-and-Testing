#!/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend_Volume
# Hypothesis: Use Ichimoku Tenkan-Kijun cross on 6h with 1d cloud filter and volume confirmation.
# The Ichimoku cloud provides dynamic support/resistance; TK cross signals momentum shifts.
# In bull markets: buy when TK crosses above in bullish cloud; sell when cross below or price leaves cloud.
# In bear markets: sell when TK crosses below in bearish cloud; cover when cross above or price leaves cloud.
# Volume confirmation reduces false signals. Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend_Volume"
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

    # Get 1d data for Ichimoku cloud (Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max() + 
                  pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max() + 
                 pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 plotted 26 periods ahead
    senkou_span_b = ((pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max() + 
                      pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    # Chikou Span (Lagging Span): close plotted 26 periods behind (not used for signals)

    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required value is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine cloud boundaries and color
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        is_bullish_cloud = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        is_bearish_cloud = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]

        if position == 0:
            # LONG: TK cross bullish + price above cloud (in bullish cloud) + volume spike
            tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
            tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
            price_above_cloud = close[i] > upper_cloud
            price_below_cloud = close[i] < lower_cloud
            
            if (tk_cross_bullish and price_above_cloud and is_bullish_cloud and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross bearish + price below cloud (in bearish cloud) + volume spike
            elif (tk_cross_bearish and price_below_cloud and is_bearish_cloud and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross bearish OR price below cloud
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or close[i] < lower_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross bullish OR price above cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or close[i] > upper_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals