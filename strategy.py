#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike_v1
Hypothesis: 6-hour Ichimoku TK cross with 1d cloud filter and volume spike confirmation.
Targets 12-37 trades/year by requiring: 1) TK cross (Tenkan/Kijun) on 6h, 2) price above/below 1d Ichimoku cloud,
3) volume > 2.0x 20-period average. Ichimoku cloud acts as dynamic support/resistance that adapts to volatility,
working in both bull and bear markets by filtering false breaks. Volume spike confirms institutional participation.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 6h Ichimoku components (Tenkan, Kijun, Senkou Span A/B)
    # Tenkan-sen: (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B: (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # 1d data for Ichimoku cloud (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Ichimoku components
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Tenkan-sen 1d: (9-period high + 9-period low) / 2
    period9_high_1d = pd.Series(df_1d_high).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(df_1d_low).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    # Kijun-sen 1d: (26-period high + 26-period low) / 2
    period26_high_1d = pd.Series(df_1d_high).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(df_1d_low).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # Senkou Span A 1d: (Tenkan + Kijun) / 2
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B 1d: (52-period high + 52-period low) / 2
    period52_high_1d = pd.Series(df_1d_high).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(df_1d_low).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Cloud top/bottom (Senkou Span A/B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # 1d Cloud top/bottom aligned
    cloud_top_1d_aligned = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom_1d_aligned = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 6h Ichimoku (52) and 1d Ichimoku (52)
    start_idx = 53
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tenkan_1d[i]) or np.isnan(kijun_1d[i]) or np.isnan(senkou_a_1d_aligned[i]) or
            np.isnan(senkou_b_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price relative to 1d Ichimoku cloud
        above_cloud = curr_close > cloud_top_1d_aligned[i]
        below_cloud = curr_close < cloud_bottom_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation
            # Long: TK cross bullish (Tenkan > Kijun) and price above 1d cloud
            tk_bullish = tenkan[i] > kijun[i]
            long_entry = tk_bullish and above_cloud and volume_confirm[i]
            
            # Short: TK cross bearish (Tenkan < Kijun) and price below 1d cloud
            tk_bearish = tenkan[i] < kijun[i]
            short_entry = tk_bearish and below_cloud and volume_confirm[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if TK cross turns bearish or price drops below 1d cloud
            if tenkan[i] < kijun[i] or not above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if TK cross turns bullish or price rises above 1d cloud
            if tenkan[i] > kijun[i] or not below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0