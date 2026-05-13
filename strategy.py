#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_Filter
Hypothesis: Ichimoku Cloud with TK cross and cloud filter from daily timeframe provides robust trend signals. 
The daily cloud acts as major support/resistance, while TK cross on 6h provides entry timing.
Works in bull markets (trend following) and bear markets (counter-trend at cloud edges).
Target: 20-50 trades/year with strict entry conditions to avoid overtrading.
"""

name = "6h_Ichimoku_Cloud_Trend_Filter"
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
    
    # Calculate Ichimoku components on 6h
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
    
    # Get daily Ichimoku for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily Tenkan and Kijun
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    d_tenkan = (pd.Series(d_high).rolling(window=9, min_periods=9).max().values + 
                pd.Series(d_low).rolling(window=9, min_periods=9).min().values) / 2
    d_kijun = (pd.Series(d_high).rolling(window=26, min_periods=26).max().values + 
               pd.Series(d_low).rolling(window=26, min_periods=26).min().values) / 2
    
    # Daily Senkou Span A and B
    d_senkou_a = (d_tenkan + d_kijun) / 2
    d_senkou_b = (pd.Series(d_high).rolling(window=52, min_periods=52).max().values + 
                  pd.Series(d_low).rolling(window=52, min_periods=52).min().values) / 2
    
    # Align daily components to 6h
    d_tenkan_aligned = align_htf_to_ltf(prices, df_1d, d_tenkan)
    d_kijun_aligned = align_htf_to_ltf(prices, df_1d, d_kijun)
    d_senkou_a_aligned = align_htf_to_ltf(prices, df_1d, d_senkou_a)
    d_senkou_b_aligned = align_htf_to_ltf(prices, df_1d, d_senkou_b)
    
    # Cloud top and bottom (for daily timeframe)
    d_cloud_top = np.maximum(d_senkou_a_aligned, d_senkou_b_aligned)
    d_cloud_bottom = np.minimum(d_senkou_a_aligned, d_senkou_b_aligned)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Need enough data for Ichimoku
        if position == 0:
            # LONG: Price above daily cloud AND TK cross bullish AND volume confirmation
            if (close[i] > d_cloud_top[i] and 
                tenkan[i] > kijun[i] and 
                tenkan[i-1] <= kijun[i-1] and  # TK cross just happened
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below daily cloud AND TK cross bearish AND volume confirmation
            elif (close[i] < d_cloud_bottom[i] and 
                  tenkan[i] < kijun[i] and 
                  tenkan[i-1] >= kijun[i-1] and  # TK cross just happened
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below daily cloud OR TK cross bearish
            if close[i] < d_cloud_top[i] or (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above daily cloud OR TK cross bullish
            if close[i] > d_cloud_bottom[i] or (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals