# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter
Hypothesis: Ichimoku Tenkan-Kijun cross (9,26) with daily cloud filter (Senkou Span A/B) and volume confirmation captures momentum with trend alignment. Works in bull/bear via cloud color (green=uptrend, red=downtrend) and avoids false signals in chop. Designed for low trade frequency on 6h timeframe.
"""

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku cloud (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d data
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2, plotted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        if position == 0:
            # LONG: Tenkan crosses above Kijun, price above cloud (green cloud), volume spike
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1] and  # cross just happened
                close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i]) and  # above cloud
                senkou_a_aligned[i] > senkou_b_aligned[i] and  # green cloud (bullish)
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan crosses below Kijun, price below cloud (red cloud), volume spike
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                  tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and  # cross just happened
                  close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i]) and  # below cloud
                  senkou_a_aligned[i] < senkou_b_aligned[i] and  # red cloud (bearish)
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan crosses below Kijun or price drops below cloud
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]) or \
               close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan crosses above Kijun or price rises above cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]) or \
               close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals