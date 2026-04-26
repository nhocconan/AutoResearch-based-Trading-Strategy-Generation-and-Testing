#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_v1
Hypothesis: Ichimoku cloud strategy on 6h with 1d trend filter. Long when price above cloud, Tenkan > Kijun, and 1d EMA50 uptrend; short when price below cloud, Tenkan < Kijun, and 1d EMA50 downtrend. Uses discrete position sizing (0.25) to minimize fee churn. Works in both bull and bear markets by following the 1d EMA50 trend direction, with Ichimoku providing timely entries/exits within the trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need warmup for Ichimoku components
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current cloud boundaries (Senkou Span A/B from 26 periods ago)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 52 for Senkou B, 26 for shift)
    start_idx = 52 + 26  # 78
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        ema_val = ema_50_1d_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(tenkan_val) or np.isnan(kijun_val) or 
            np.isnan(cloud_top_val) or np.isnan(cloud_bottom_val) or 
            np.isnan(ema_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Ichimoku bullish: price above cloud, Tenkan > Kijun
        ichimoku_bullish = (close_val > cloud_top_val) and (tenkan_val > kijun_val)
        # Ichimoku bearish: price below cloud, Tenkan < Kijun
        ichimoku_bearish = (close_val < cloud_bottom_val) and (tenkan_val < kijun_val)
        
        # 1d trend filter
        trend_up = ema_val is not None and not np.isnan(ema_val) and close_val > ema_val
        trend_down = ema_val is not None and not np.isnan(ema_val) and close_val < ema_val
        
        # Long logic: Ichimoku bullish + 1d uptrend
        long_condition = ichimoku_bullish and trend_up
        # Short logic: Ichimoku bearish + 1d downtrend
        short_condition = ichimoku_bearish and trend_down
        
        # Exit logic: 
        # Long exit: price crosses below cloud OR Tenkan < Kijun (trend weakness)
        long_exit = (position == 1 and (close_val < cloud_top_val or tenkan_val < kijun_val))
        # Short exit: price crosses above cloud OR Tenkan > Kijun (trend weakness)
        short_exit = (position == -1 and (close_val > cloud_bottom_val or tenkan_val > kijun_val))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_v1"
timeframe = "6h"
leverage = 1.0