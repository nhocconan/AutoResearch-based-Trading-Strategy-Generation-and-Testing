#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Breakout_1dTrend_Volume
# Hypothesis: Use 1d Ichimoku cloud and 6h Tenkan/Kijun cross for breakout entries with volume confirmation.
# Long when Tenkan crosses above Kijun AND price above cloud in uptrend (Tenkan > Kijun on 1d).
# Short when Tenkan crosses below Kijun AND price below cloud in downtrend (Tenkan < Kijun on 1d).
# Exit when price crosses back into the cloud or Tenkan/Kijun cross reverses.
# Designed for moderate trade frequency (50-150 total trades over 4 years) with clear entry/exit rules.

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max()
    tenkan_9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()
    tenkan_sen = ((tenkan_9_high + tenkan_9_low) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max()
    kijun_26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()
    kijun_sen = ((kijun_26_high + kijun_26_low) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low) / 2, shifted 26 periods ahead
    senkou_52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max()
    senkou_52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min()
    senkou_b = ((senkou_52_high + senkou_52_low) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Ichimoku calculation window
        # Skip if any required value is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Determine 1d trend: Tenkan > Kijun = uptrend, Tenkan < Kijun = downtrend
        trend_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        trend_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i]

        if position == 0:
            # LONG: Tenkan crosses above Kijun AND price above cloud AND uptrend + volume spike
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1] and  # Cross just happened
                close[i] > cloud_top and
                trend_up and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan crosses below Kijun AND price below cloud AND downtrend + volume spike
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and  # Cross just happened
                  close[i] < cloud_bottom and
                  trend_down and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back into cloud OR Tenkan/Kijun cross reverses
            if (close[i] <= cloud_top and close[i] >= cloud_bottom) or \
               (tenkan_sen_aligned[i] < kijun_sen_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back into cloud OR Tenkan/Kijun cross reverses
            if (close[i] <= cloud_top and close[i] >= cloud_bottom) or \
               (tenkan_sen_aligned[i] > kijun_sen_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals