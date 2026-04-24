#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud strategy with 1d trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) for trend and cloud filter.
- Entry: Long when price > cloud AND Tenkan > Kijun (bullish) AND volume > 1.5 * volume MA(20).
         Short when price < cloud AND Tenkan < Kijun (bearish) AND volume > 1.5 * volume MA(20).
- Exit: Opposite Ichimoku TK cross (Tenkan crosses below/above Kijun) OR price re-enters cloud.
- Signal size: 0.25 discrete for drawdown control.
Designed to work in both bull and bear markets via cloud filter (trend) and TK cross (momentum).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Senkou Span B (26+26)
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align HTF Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate volume MA(20) for confirmation (using 6h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 52, 20)  # Need enough bars for Ichimoku (52) and volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Bullish conditions: price above cloud AND Tenkan > Kijun
            bullish = (curr_close > cloud_top) and (tenkan_aligned[i] > kijun_aligned[i])
            # Bearish conditions: price below cloud AND Tenkan < Kijun
            bearish = (curr_close < cloud_bottom) and (tenkan_aligned[i] < kijun_aligned[i])
            
            # Long entry
            if bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry
            elif bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun OR price re-enters cloud (below cloud top)
            if (tenkan_aligned[i] < kijun_aligned[i]) or (curr_close < cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun OR price re-enters cloud (above cloud bottom)
            if (tenkan_aligned[i] > kijun_aligned[i]) or (curr_close > cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TKCross_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0