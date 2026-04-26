#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: On 6h timeframe, enter long when Tenkan-sen crosses above Kijun-sen AND price is above 1d Ichimoku cloud (bullish regime), short when Tenkan crosses below Kijun AND price is below 1d cloud. Uses volume confirmation (>1.3x 20-period MA) to filter false breaks. Ichimoku cloud from 1d provides multi-timeframe trend filter that works in both bull and bear markets by only taking trades aligned with higher timeframe regime. Discrete position sizing (0.25) minimizes fee churn. Target: 12-37 trades/year (50-150 total over 4 years).
"""

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
    
    # Get 1d data for Ichimoku cloud (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Senkou B
        return np.zeros(n)
    
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
    
    # Get 1d Ichimoku cloud for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Tenkan-sen (9-period)
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max().values + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).min().values) / 2
    
    # 1d Kijun-sen (26-period)
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max().values + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min().values) / 2
    
    # 1d Senkou Span A
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # 1d Senkou Span B (52-period)
    senkou_b_1d = (pd.Series(high_1d).rolling(window=52, min_periods=52).max().values + 
                   pd.Series(low_1d).rolling(window=52, min_periods=52).min().values) / 2
    
    # 1d Cloud boundaries: max/min of Senkou A/B
    upper_cloud_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    lower_cloud_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 1d cloud to 6h timeframe (already completed 1d bar)
    upper_cloud_aligned = align_htf_to_ltf(prices, df_1d, upper_cloud_1d)
    lower_cloud_aligned = align_htf_to_ltf(prices, df_1d, lower_cloud_1d)
    
    # TK Cross signals
    tk_cross_above = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    tk_cross_below = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
    
    # Price relative to 1d cloud
    price_above_cloud = close > upper_cloud_aligned
    price_below_cloud = close < lower_cloud_aligned
    
    # Volume confirmation: volume > 1.3x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for Kijun, 20 for volume MA)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_cloud_aligned[i]) or np.isnan(lower_cloud_aligned[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: TK cross above + price above 1d cloud + volume spike
            if (tk_cross_above[i] and price_above_cloud[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross below + price below 1d cloud + volume spike
            elif (tk_cross_below[i] and price_below_cloud[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross below (trend change) OR price breaks below 1d cloud (regime change)
            if tk_cross_below[i] or not price_above_cloud[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross above (trend change) OR price breaks above 1d cloud (regime change)
            if tk_cross_above[i] or not price_below_cloud[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0