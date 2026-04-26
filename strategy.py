#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1wTrend_VolumeFilter_v1
Hypothesis: Ichimoku TK cross with weekly trend filter and volume confirmation on 6h timeframe. 
Weekly trend determined by price vs Kumo cloud (senkou span A/B) from 1w data. 
Enter long when TK cross bullish + price above weekly cloud + volume > 1.5x average.
Enter short when TK cross bearish + price below weekly cloud + volume > 1.5x average.
Exit when TK cross reverses or volume drops below average.
Designed for low trade frequency (target 12-30/year) to work in both bull (trend following) and bear (counter-trend at extremes) markets.
Discrete position sizing (0.25) minimizes fee churn. Uses 1w HTF for major trend filter to avoid whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Ichimoku and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 52 weeks for proper Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on weekly data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1w['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1w['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1w['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1w['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2.0)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = df_1w['close'].values
    
    # Align Ichimoku components to 6h timeframe (wait for weekly bar to close)
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    chikou_aligned = align_htf_to_ltf(prices, df_1w, chikou_span)
    
    # Calculate volume filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku periods (52) and volume MA (20)
    start_idx = max(52, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        vol_filt = volume_filter[i]
        
        # Determine Kumo cloud boundaries (top and bottom of cloud)
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # TK cross signals
        tk_bullish = tenkan > kijun
        tk_bearish = tenkan < kijun
        
        # Price relative to cloud
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        if position == 0:
            # Long: bullish TK cross + price above cloud + volume filter
            long_signal = tk_bullish and price_above_cloud and vol_filt
            
            # Short: bearish TK cross + price below cloud + volume filter
            short_signal = tk_bearish and price_below_cloud and vol_filt
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross turns bearish OR volume filter fails
            if not (tk_bullish and vol_filt):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross turns bullish OR volume filter fails
            if not (tk_bearish and vol_filt):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1wTrend_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0