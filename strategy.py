#!/usr/bin/env python3
"""
4h_ichimoku_cloud_bounce_v1
Hypothesis: Uses Ichimoku Cloud (9/26/52) on 4h with daily trend filter to enter on pullbacks to the Kumo (cloud) in trending markets. Long when price above cloud and Tenkan > Kijun, short when price below cloud and Tenkan < Kijun. Uses daily EMA200 for higher timeframe trend filter to avoid counter-trend trades in strong trends. Designed to work in both bull (trend continuation on pullbacks) and bear (counter-trend bounces within larger trend) markets by following higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ichimoku_cloud_bounce_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku Cloud components (9, 26, 52)
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
    
    # Daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a[i], senkou_b[i])
        lower_cloud = min(senkou_a[i], senkou_b[i])
        
        if position == 1:  # Long position
            # Exit: Price falls below cloud or trend changes
            if close[i] < lower_cloud or close[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price rises above cloud or trend changes
            if close[i] > upper_cloud or close[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Price above cloud, Tenkan > Kijun, and above daily EMA200 (uptrend)
            if (close[i] > upper_cloud and 
                tenkan[i] > kijun[i] and 
                close[i] > ema_200_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: Price below cloud, Tenkan < Kijun, and below daily EMA200 (downtrend)
            elif (close[i] < lower_cloud and 
                  tenkan[i] < kijun[i] and 
                  close[i] < ema_200_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals