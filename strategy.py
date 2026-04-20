#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_v1
Hypothesis: Use daily Ichimoku cloud as trend filter and TK cross on 6h for entry.
- TK cross (Tenkan/Kijun) on 6h provides timely entry signals
- Daily cloud (Senkou Span A/B) acts as trend filter: only go long when price above cloud, short when below
- Avoids counter-trend trades in strong trends, works in both bull/bear markets
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: tenkan, kijun, senkou_a, senkou_b, chikou"""
    n = len(high)
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9 = 9
    for i in range(period9 - 1, n):
        high9 = np.max(high[i - period9 + 1:i + 1])
        low9 = np.min(low[i - period9 + 1:i + 1])
        tenkan[i] = (high9 + low9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26 = 26
    for i in range(period26 - 1, n):
        high26 = np.max(high[i - period26 + 1:i + 1])
        low26 = np.min(low[i - period26 + 1:i + 1])
        kijun[i] = (high26 + low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    for i in range(n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52 = 52
    for i in range(period52 - 1, n):
        high52 = np.max(high[i - period52 + 1:i + 1])
        low52 = np.min(low[i - period52 + 1:i + 1])
        senkou_b[i] = (high52 + low52) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data ONCE before loop for Ichimoku
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 60:
        return np.zeros(n)
    
    # Calculate Ichimoku on daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    tenkan_daily, kijun_daily, senkou_a_daily, senkou_b_daily = calculate_ichimoku(high_daily, low_daily, close_daily)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_daily_aligned = align_htf_to_ltf(prices, df_daily, tenkan_daily)
    kijun_daily_aligned = align_htf_to_ltf(prices, df_daily, kijun_daily)
    senkou_a_daily_aligned = align_htf_to_ltf(prices, df_daily, senkou_a_daily)
    senkou_b_daily_aligned = align_htf_to_ltf(prices, df_daily, senkou_b_daily)
    
    # Calculate TK cross on 6h timeframe (using same Ichimoku calculation but on 6h data)
    tenkan_6h, kijun_6h, _, _ = calculate_ichimoku(high, low, close)
    tk_cross = tenkan_6h - kijun_6h  # Positive when Tenkan > Kijun (bullish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_daily_aligned[i]) or np.isnan(kijun_daily_aligned[i]) or
            np.isnan(senkou_a_daily_aligned[i]) or np.isnan(senkou_b_daily_aligned[i]) or
            np.isnan(tk_cross[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_daily_aligned[i], senkou_b_daily_aligned[i])
        lower_cloud = np.minimum(senkou_a_daily_aligned[i], senkou_b_daily_aligned[i])
        
        if position == 0:
            # Long: TK cross bullish (Tenkan > Kijun) AND price above cloud
            if tk_cross[i] > 0 and close[i] > upper_cloud:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish (Tenkan < Kijun) AND price below cloud
            elif tk_cross[i] < 0 and close[i] < lower_cloud:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross bearish OR price drops below cloud
            if tk_cross[i] < 0 or close[i] < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross bullish OR price rises above cloud
            if tk_cross[i] > 0 or close[i] > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals