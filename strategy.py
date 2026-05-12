#!/usr/bin/env python3
name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Chikou Span (Lagging Span): not used for signals (requires future data)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # need enough data for Senkou B
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        # Cloud top = max(senkou_a, senkou_b), cloud bottom = min(senkou_a, senkou_b)
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Long conditions: TK cross bullish, price above cloud, 1d uptrend
            if (tenkan[i] > kijun[i] and  # TK cross bullish
                close[i] > cloud_top and  # price above cloud
                close[i] > ema50_1d_aligned[i]):  # 1d uptrend
                signals[i] = 0.25
                position = 1
            # Short conditions: TK cross bearish, price below cloud, 1d downtrend
            elif (tenkan[i] < kijun[i] and  # TK cross bearish
                  close[i] < cloud_bottom and  # price below cloud
                  close[i] < ema50_1d_aligned[i]):  # 1d downtrend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when TK cross bearish or price drops below cloud
            if (tenkan[i] < kijun[i] or 
                close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when TK cross bullish or price rises above cloud
            if (tenkan[i] > kijun[i] or 
                close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals