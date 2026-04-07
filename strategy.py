#!/usr/bin/env python3
"""
6H Ichimoku Cloud with Weekly Filter - Trend Following Strategy
Hypothesis: Ichimoku provides robust trend signals with cloud acting as dynamic support/resistance.
In trending markets (bull or bear), price stays above/below cloud with TK cross confirming direction.
Weekly trend filter ensures we only trade in direction of higher timeframe trend, reducing whipsaws.
Uses weekly trend to filter Ichimoku signals, targeting 20-40 trades/year with 0.25 position size.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_weekly_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:  # Need enough data for Ichimoku (52 periods)
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Ichimoku Components ===
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
    
    # === Weekly Trend Filter (EMA 50) ===
    df_weekly = get_htf_data(prices, '1w')
    ema_weekly = pd.Series(df_weekly['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if any values are NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR TK cross turns bearish
            if close[i] < cloud_bottom or tenkan[i] < kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR TK cross turns bullish
            if close[i] > cloud_top or tenkan[i] > kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need weekly trend alignment
            if ema_weekly_aligned[i] <= ema_weekly_aligned[i-1]:
                # Weekly trend not bullish enough for long
                pass
            elif ema_weekly_aligned[i] >= ema_weekly_aligned[i-1]:
                # Weekly trend not bearish enough for short
                pass
            else:
                signals[i] = 0.0
                continue
            
            # Entry conditions
            # Long: Price above cloud + bullish TK cross + weekly uptrend
            if (close[i] > cloud_top and 
                tenkan[i] > kijun[i] and 
                ema_weekly_aligned[i] > ema_weekly_aligned[i-1]):
                position = 1
                signals[i] = 0.25
            # Short: Price below cloud + bearish TK cross + weekly downtrend
            elif (close[i] < cloud_bottom and 
                  tenkan[i] < kijun[i] and 
                  ema_weekly_aligned[i] < ema_weekly_aligned[i-1]):
                position = -1
                signals[i] = -0.25
    
    return signals