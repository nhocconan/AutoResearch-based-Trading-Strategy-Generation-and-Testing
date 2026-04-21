#!/usr/bin/env python3
"""
6h_1d_Ichimoku_TK_Cross_CloudFilter_V1
Hypothesis: Ichimoku's Tenkan-sen/Kijun-sen cross with cloud filter provides high-probability trend entries. Daily Ichimoku cloud filters the 6h TK cross for major trend direction. TK cross above cloud = strong uptrend; TK cross below cloud = strong downtrend. Uses Tenkan-sen (9-period) and Kijun-sen (26-period) on 6m, with Senkou Span A/B calculated from daily high/low. Designed for low trade frequency (target: 12-37/year) on 6h timeframe to minimize fee drag. Works in bull markets via trend continuation and bear markets via counter-trend reversals at cloud boundaries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Ichimoku cloud
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 52:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Ichimoku components on daily
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    
    # Tenkan-sen (9-period)
    high_9 = pd.Series(high_daily).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_daily).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2.0
    
    # Kijun-sen (26-period)
    high_26 = pd.Series(high_daily).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_daily).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2.0
    
    # Senkou Span A
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (52-period)
    high_52 = pd.Series(high_daily).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_daily).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2.0
    
    # Align daily Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_daily, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_daily, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_daily, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_daily, senkou_span_b)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Ichimoku components for TK cross
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_6h = (high_9_6h + low_9_6h) / 2.0
    
    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen_6h = (high_26_6h + low_26_6h) / 2.0
    
    # TK cross signals
    tk_cross_above = tenkan_sen_6h > kijun_sen_6h
    tk_cross_below = tenkan_sen_6h < kijun_sen_6h
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(26, n):
        # Skip if NaN in critical values
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        tenkan = tenkan_sen_6h[i]
        kijun = kijun_sen_6h[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        vol_ok = volume_filter[i]
        
        # Determine cloud relationship
        price_above_cloud = price > cloud_top_val
        price_below_cloud = price < cloud_bottom_val
        price_in_cloud = (price >= cloud_bottom_val) and (price <= cloud_top_val)
        
        # TK cross signals
        tk_bullish = tk_cross_above[i]
        tk_bearish = tk_cross_below[i]
        
        if position == 0:
            # Long: TK bullish cross AND price above cloud (strong uptrend)
            if tk_bullish and price_above_cloud and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: TK bearish cross AND price below cloud (strong downtrend)
            elif tk_bearish and price_below_cloud and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: TK bearish cross OR price drops below cloud bottom
            if tk_bearish or price < cloud_bottom_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK bullish cross OR price rises above cloud top
            if tk_bullish or price > cloud_top_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Ichimoku_TK_Cross_CloudFilter_V1"
timeframe = "6h"
leverage = 1.0