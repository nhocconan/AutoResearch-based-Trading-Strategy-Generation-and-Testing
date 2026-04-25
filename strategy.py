#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1wTrend_Filter_v2
Hypothesis: Trade Ichimoku TK cross signals on 6h with weekly trend filter. In bullish weekly trend (price above weekly cloud), buy when Tenkan crosses above Kijun; in bearish weekly trend (price below weekly cloud), sell when Tenkan crosses below Kijun. Uses discrete position sizing (0.25) to minimize fee drag and target ~15-30 trades/year. Weekly trend filter reduces false signals in choppy markets and works in both bull and bear regimes by following higher timeframe structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
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
    
    # Weekly trend filter: price relative to weekly cloud
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly Ichimoku cloud
    weekly_period9_high = pd.Series(weekly_high).rolling(window=9, min_periods=9).max().values
    weekly_period9_low = pd.Series(weekly_low).rolling(window=9, min_periods=9).min().values
    weekly_tenkan = (weekly_period9_high + weekly_period9_low) / 2
    
    weekly_period26_high = pd.Series(weekly_high).rolling(window=26, min_periods=26).max().values
    weekly_period26_low = pd.Series(weekly_low).rolling(window=26, min_periods=26).min().values
    weekly_kijun = (weekly_period26_high + weekly_period26_low) / 2
    
    weekly_senkou_a = (weekly_tenkan + weekly_kijun) / 2
    weekly_period52_high = pd.Series(weekly_high).rolling(window=52, min_periods=52).max().values
    weekly_period52_low = pd.Series(weekly_low).rolling(window=52, min_periods=52).min().values
    weekly_senkou_b = (weekly_period52_high + weekly_period52_low) / 2
    
    # Weekly cloud boundaries (shifted forward 26 periods)
    weekly_senkou_a_shifted = np.roll(weekly_senkou_a, 26)
    weekly_senkou_b_shifted = np.roll(weekly_senkou_b, 26)
    weekly_senkou_a_shifted[:26] = np.nan
    weekly_senkou_b_shifted[:26] = np.nan
    
    weekly_cloud_top = np.maximum(weekly_senkou_a_shifted, weekly_senkou_b_shifted)
    weekly_cloud_bottom = np.minimum(weekly_senkou_a_shifted, weekly_senkou_b_shifted)
    
    # Align weekly cloud to 6h timeframe
    weekly_cloud_top_aligned = align_htf_to_ltf(prices, df_1w, weekly_cloud_top, additional_delay_bars=1)
    weekly_cloud_bottom_aligned = align_htf_to_ltf(prices, df_1w, weekly_cloud_bottom, additional_delay_bars=1)
    
    # TK cross signals
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Handle first value
    tk_cross_up[0] = False
    tk_cross_down[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku calculations
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if weekly cloud data not ready
        if (np.isnan(weekly_cloud_top_aligned[i]) or 
            np.isnan(weekly_cloud_bottom_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly trend: price above/below weekly cloud
        price_above_cloud = close[i] > weekly_cloud_top_aligned[i]
        price_below_cloud = close[i] < weekly_cloud_bottom_aligned[i]
        
        if position == 0:
            # Look for TK cross signals with weekly trend alignment
            long_signal = tk_cross_up[i] and price_above_cloud
            short_signal = tk_cross_down[i] and price_below_cloud
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when TK cross down or price falls below weekly cloud
            exit_signal = tk_cross_down[i] or not price_above_cloud
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when TK cross up or price rises above weekly cloud
            exit_signal = tk_cross_up[i] or not price_below_cloud
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1wTrend_Filter_v2"
timeframe = "6h"
leverage = 1.0