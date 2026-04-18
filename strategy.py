#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d Trend Filter
Long: Price above Kumo (cloud) + Tenkan > Kijun + 1d EMA50 up
Short: Price below Kumo + Tenkan < Kijun + 1d EMA50 down
Ichimoku provides multi-line support/resistance and momentum signals.
The 1d EMA50 filter ensures alignment with higher timeframe trend.
Designed for 6h timeframe to capture sustained trends while avoiding whipsaws.
Target: 60-120 total trades over 4 years (15-30/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2).shift(26)
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku on 6h
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d EMA slope for trend filter
    ema_slope = np.diff(ema_50_1d_aligned, prepend=ema_50_1d_aligned[0])
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 60  # need Ichimoku calculations
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_slope[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        # Kumo (cloud) top and bottom
        kumo_top = max(senkou_a[i], senkou_b[i])
        kumo_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Long: Price above Kumo + Tenkan > Kijun + 1d EMA50 up
            if (price > kumo_top and 
                tenkan[i] > kijun[i] and 
                ema_slope[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: Price below Kumo + Tenkan < Kijun + 1d EMA50 down
            elif (price < kumo_bottom and 
                  tenkan[i] < kijun[i] and 
                  ema_slope[i] < 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below Kumo bottom OR Tenkan < Kijun
            if (price < kumo_bottom) or (tenkan[i] < kijun[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above Kumo top OR Tenkan > Kijun
            if (price > kumo_top) or (tenkan[i] > kijun[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0