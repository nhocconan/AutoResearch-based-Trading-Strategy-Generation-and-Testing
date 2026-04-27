#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrend_VolumeConfirm
Hypothesis: Ichimoku TK cross with cloud filter from 1d timeframe on 6h chart. Uses TK cross (Tenkan/Kijun) for entry timing, 1d cloud color for trend filter, and volume confirmation for validity. Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year). Works in both bull and bear markets by following 1d Ichimoku trend while using TK cross for precise entries. Avoids overtrading with tight entry conditions.
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
    
    # Get 1d data for Ichimoku cloud and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((max_high_52 + min_low_52) / 2)
    
    # Align 1d indicators to 6h timeframe (completed bars only)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Cloud color: green (bullish) when Senkou A > Senkou B, red (bearish) when Senkou A < Senkou B
    cloud_bullish = senkou_a_aligned > senkou_b_aligned
    cloud_bearish = senkou_a_aligned < senkou_b_aligned
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Ichimoku components (52) + volume avg (20)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: TK cross with cloud filter and volume confirmation
            # Bullish TK cross: Tenkan crosses above Kijun
            # Bearish TK cross: Tenkan crosses below Kijun
            bullish_cross = (tenkan_val > kijun_val) and (i > start_idx and tenkan_aligned[i-1] <= kijun_aligned[i-1])
            bearish_cross = (tenkan_val < kijun_val) and (i > start_idx and tenkan_aligned[i-1] >= kijun_aligned[i-1])
            
            # Long: bullish TK cross AND bullish cloud AND volume confirmation
            long_condition = bullish_cross and cloud_bullish[i] and vol_conf
            # Short: bearish TK cross AND bearish cloud AND volume confirmation
            short_condition = bearish_cross and cloud_bearish[i] and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: TK cross turns bearish OR cloud turns bearish
            bearish_cross_exit = (tenkan_val < kijun_val) and (i > start_idx and tenkan_aligned[i-1] >= kijun_aligned[i-1])
            cloud_turns_bearish = cloud_bearish[i]
            
            if bearish_cross_exit or cloud_turns_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: TK cross turns bullish OR cloud turns bullish
            bullish_cross_exit = (tenkan_val > kijun_val) and (i > start_idx and tenkan_aligned[i-1] <= kijun_aligned[i-1])
            cloud_turns_bullish = cloud_bullish[i]
            
            if bullish_cross_exit or cloud_turns_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0