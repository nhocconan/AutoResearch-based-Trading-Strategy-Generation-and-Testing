#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter
Hypothesis: Tenkan-sen/Kijun-sen cross on 6h chart with 1d Ichimoku cloud filter captures momentum shifts while avoiding trades against the higher timeframe trend. Works in bull/bear by using cloud color (green/red) as regime filter.
"""

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter"
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
    
    # Get 1d data for Ichimoku cloud (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # TK cross signals on 6h
    tk_cross_up = (tenkan_sen_aligned > kijun_sen_aligned) & (tenkan_sen_aligned <= kijun_sen_aligned + 1e-10)  # Avoid exact equality issues
    tk_cross_down = (tenkan_sen_aligned < kijun_sen_aligned) & (tenkan_sen_aligned >= kijun_sen_aligned - 1e-10)
    
    # Cloud color: green when Senkou Span A > Senkou Span B (bullish), red when A < B (bearish)
    cloud_green = senkou_span_a_aligned > senkou_span_b_aligned
    cloud_red = senkou_span_a_aligned < senkou_span_b_aligned
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if position == 0:
            # LONG: TK cross up in bullish cloud with volume spike
            if tk_cross_up[i] and cloud_green[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross down in bearish cloud with volume spike
            elif tk_cross_down[i] and cloud_red[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross down or cloud turns bearish
            if tk_cross_down[i] or (not cloud_green[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross up or cloud turns bullish
            if tk_cross_up[i] or cloud_green[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals