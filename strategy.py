#!/usr/bin/env python3
# Hypothesis: 6h Ichimoku Cloud with TK Cross and 1d trend filter for BTC/ETH.
# Uses 1d Ichimoku trend (price above/below cloud) to filter 6h TK cross signals.
# Ichimoku components: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (52-period displacement).
# Entry: Long when TK crosses above AND price > 1d cloud top; Short when TK crosses below AND price < 1d cloud bottom.
# Exit: Opposite TK cross or price re-enters cloud.
# Designed for low frequency (target 50-150 total trades over 4 years) with trend-following edge in both bull/bear markets.

name = "6h_Ichimoku_TK_Cross_1dCloudFilter_v1"
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
    
    # Calculate 1d Ichimoku components for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).mean() +
                 pd.Series(low_1d).rolling(window=9, min_periods=9).mean()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).mean() +
                pd.Series(low_1d).rolling(window=26, min_periods=26).mean()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).mean() +
                    pd.Series(low_1d).rolling(window=52, min_periods=52).mean()) / 2).shift(26)
    
    # Cloud top/bottom: max/min of Senkou Span A/B
    cloud_top_1d = np.maximum(senkou_a_1d.values, senkou_b_1d.values)
    cloud_bottom_1d = np.minimum(senkou_a_1d.values, senkou_b_1d.values)
    
    # Align 1d Ichimoku components to 6h timeframe (wait for 1d bar to close)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    
    # Calculate 6h Ichimoku components for entry signals
    tenkan_6h = (pd.Series(high).rolling(window=9, min_periods=9).mean() +
                 pd.Series(low).rolling(window=9, min_periods=9).mean()) / 2
    kijun_6h = (pd.Series(high).rolling(window=26, min_periods=26).mean() +
                pd.Series(low).rolling(window=26, min_periods=26).mean()) / 2
    
    # TK Cross signals: Tenkan-sen crossing Kijun-sen
    tk_cross_above = (tenkan_6h > kijun_6h) & (tenkan_6h.shift(1) <= kijun_6h.shift(1))
    tk_cross_below = (tenkan_6h < kijun_6h) & (tenkan_6h.shift(1) >= kijun_6h.shift(1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(cloud_top_aligned[i]) or 
            np.isnan(cloud_bottom_aligned[i]) or 
            np.isnan(tenkan_6h.iloc[i]) or 
            np.isnan(kijun_6h.iloc[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TK cross above AND price > 1d cloud top
            if tk_cross_above.iloc[i] and close[i] > cloud_top_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross below AND price < 1d cloud bottom
            elif tk_cross_below.iloc[i] and close[i] < cloud_bottom_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross below OR price < 1d cloud top (re-enters cloud)
            if tk_cross_below.iloc[i] or close[i] < cloud_top_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross above OR price > 1d cloud bottom (re-enters cloud)
            if tk_cross_above.iloc[i] or close[i] > cloud_bottom_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals