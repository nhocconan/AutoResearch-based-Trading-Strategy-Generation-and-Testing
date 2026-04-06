#!/usr/bin/env python3
"""
6h Ichimoku Cloud with Daily Filter and Volume Confirmation
Hypothesis: Ichimoku TK cross signals aligned with daily trend (price above/below Kumo) 
capture trend continuation in both bull and bear markets. Daily filter avoids counter-trend 
trades, volume confirms momentum. Ichimoku provides dynamic support/resistance for exits.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_daily_filter_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan = np.full(n, np.nan)
    for i in range(9, n):
        tenkan[i] = (np.max(high[i-9:i]) + np.min(low[i-9:i])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun = np.full(n, np.nan)
    for i in range(26, n):
        kijun[i] = (np.max(high[i-26:i]) + np.min(low[i-26:i])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = np.full(n, np.nan)
    for i in range(26, n):
        senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_b = np.full(n, np.nan)
    for i in range(52, n):
        senkou_b[i] = (np.max(high[i-52:i]) + np.min(low[i-52:i])) / 2
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily Ichimoku for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    tenkan_1d = np.full(len(close_1d), np.nan)
    kijun_1d = np.full(len(close_1d), np.nan)
    senkou_a_1d = np.full(len(close_1d), np.nan)
    senkou_b_1d = np.full(len(close_1d), np.nan)
    
    for i in range(9, len(close_1d)):
        tenkan_1d[i] = (np.max(high_1d[i-9:i]) + np.min(low_1d[i-9:i])) / 2
    for i in range(26, len(close_1d)):
        kijun_1d[i] = (np.max(high_1d[i-26:i]) + np.min(low_1d[i-26:i])) / 2
    for i in range(26, len(close_1d)):
        senkou_a_1d[i] = (tenkan_1d[i] + kijun_1d[i]) / 2
    for i in range(52, len(close_1d)):
        senkou_b_1d[i] = (np.max(high_1d[i-52:i]) + np.min(low_1d[i-52:i])) / 2
    
    # Daily trend: 1 if price > Kumo (above both spans), -1 if price < Kumo (below both spans)
    kumos_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumos_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    trend_1d = np.where(close_1d > kumos_top_1d, 1, np.where(close_1d < kumos_bottom_1d, -1, 0))
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(52, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or \
           np.isnan(senkou_b[i]) or np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: TK cross down OR price breaks below Kumo
            if (tenkan[i] < kijun[i] or 
                close[i] < senkou_a[i] or 
                close[i] < senkou_b[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross up OR price breaks above Kumo
            if (tenkan[i] > kijun[i] or 
                close[i] > senkou_a[i] or 
                close[i] > senkou_b[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: TK cross up in uptrend with volume
            if (tenkan[i] > kijun[i] and
                trend_1d_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: TK cross down in downtrend with volume
            elif (tenkan[i] < kijun[i] and
                  trend_1d_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals