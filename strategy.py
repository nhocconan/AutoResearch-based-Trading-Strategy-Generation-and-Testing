#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: Trade Ichimoku Tenkan/Kijun cross with cloud filter from 1d timeframe. 
In bull markets: long when price above cloud and TK cross bullish. 
In bear markets: short when price below cloud and TK cross bearish.
Uses 1d trend filter to align with higher timeframe direction, reducing whipsaws.
Target: 15-30 trades/year on 6f (60-120 total over 4 years).
"""

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
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Ichimoku Components (9, 26, 52 periods) on 6h data ===
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
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For cloud at time t, we use Senkou values from t-26
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values are invalid due to shift
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # === 1d Trend Filter (EMA 50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Ichimoku calculations)
    start_idx = 80
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(ema50_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Bullish conditions: price above cloud, TK cross bullish, 1d uptrend
        bullish = (close[i] > cloud_top[i] and 
                   tenkan[i] > kijun[i] and 
                   close[i] > ema50_6h[i])
        
        # Bearish conditions: price below cloud, TK cross bearish, 1d downtrend
        bearish = (close[i] < cloud_bottom[i] and 
                   tenkan[i] < kijun[i] and 
                   close[i] < ema50_6h[i])
        
        if position == 0:
            if bullish:
                signals[i] = 0.25
                position = 1
            elif bearish:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price drops below cloud or TK cross turns bearish
            if close[i] < cloud_bottom[i] or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above cloud or TK cross turns bullish
            if close[i] > cloud_top[i] or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals