#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeSpike
Hypothesis: Ichimoku cloud breakout on 6h with weekly trend filter and volume confirmation. 
Weekly trend (price above/below weekly cloud) filters for higher probability breaks of the 6h Ichimoku cloud.
Volume spike (>2.0x 20-bar MA) confirms momentum. Designed for 6h timeframe to work in both bull and bear markets:
- In bull markets: weekly uptrend + 6h cloud breakout long = momentum continuation
- In bear markets: weekly downtrend + 6h cloud breakdown short = trend following
Ichimoku provides objective support/resistance (cloud) and momentum (TK cross) in one indicator.
Target: 12-25 trades/year (50-100 total over 4 years).
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
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly Ichimoku cloud for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Ichimoku calculation (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align weekly Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    # Load daily data for 6h Ichimoku calculation (more responsive)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 6h Ichimoku components using daily data (standard practice)
    # Tenkan-sen (6h): (9-period high + 9-period low)/2 from daily
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen_6h = (period9_high_1d + period9_low_1d) / 2
    
    # Kijun-sen (6h): (26-period high + 26-period low)/2 from daily
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen_6h = (period26_high_1d + period26_low_1d) / 2
    
    # Senkou Span A (6h): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a_6h = ((tenkan_sen_6h + kijun_sen_6h) / 2)
    # Senkou Span B (6h): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_6h = ((period52_high_1d + period52_low_1d) / 2)
    
    # Align 6h Ichimoku components to 6h timeframe
    tenkan_sen_6h_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_6h)
    kijun_sen_6h_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_6h)
    senkou_a_6h_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_6h)
    senkou_b_6h_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_6h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (26 for Ichimoku, 20 for volume)
    start_idx = max(26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_6h_aligned[i]) or 
            np.isnan(kijun_sen_6h_aligned[i]) or 
            np.isnan(senkou_a_6h_aligned[i]) or 
            np.isnan(senkou_b_6h_aligned[i]) or
            np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        
        # 6h Ichimoku values
        tenkan_6h = tenkan_sen_6h_aligned[i]
        kijun_6h = kijun_sen_6h_aligned[i]
        senkou_a_6h = senkou_a_6h_aligned[i]
        senkou_b_6h = senkou_b_6h_aligned[i]
        
        # Weekly Ichimoku values for trend filter
        tenkan_w = tenkan_sen_aligned[i]
        kijun_w = kijun_sen_aligned[i]
        senkou_a_w = senkou_a_aligned[i]
        senkou_b_w = senkou_b_aligned[i]
        
        # Determine 6h cloud boundaries (Senkou Span A/B)
        upper_cloud_6h = max(senkou_a_6h, senkou_b_6h)
        lower_cloud_6h = min(senkou_a_6h, senkou_b_6h)
        
        # Determine weekly cloud boundaries and trend
        upper_cloud_w = max(senkou_a_w, senkou_b_w)
        lower_cloud_w = min(senkou_a_w, senkou_b_w)
        
        # Weekly trend: price above weekly cloud = bullish, below = bearish
        weekly_bullish = close_val > upper_cloud_w
        weekly_bearish = close_val < lower_cloud_w
        
        # 6h TK cross: Tenkan-sen crossing above/below Kijun-sen
        tk_cross_bullish = tenkan_6h > kijun_6h
        tk_cross_bearish = tenkan_6h < kijun_6h
        
        # Price relative to 6h cloud
        price_above_cloud_6h = close_val > upper_cloud_6h
        price_below_cloud_6h = close_val < lower_cloud_6h
        
        # Entry conditions: Ichimoku signals aligned with weekly trend + volume spike
        long_entry = (price_above_cloud_6h and tk_cross_bullish and weekly_bullish and volume_spike[i])
        short_entry = (price_below_cloud_6h and tk_cross_bearish and weekly_bearish and volume_spike[i])
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price re-enters cloud or TK cross turns bearish
            if (close_val < upper_cloud_6h and close_val > lower_cloud_6h) or not tk_cross_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit when price re-enters cloud or TK cross turns bullish
            if (close_val < upper_cloud_6h and close_val > lower_cloud_6h) or not tk_cross_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0