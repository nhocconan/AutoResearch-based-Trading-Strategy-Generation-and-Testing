#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrend_v2
Hypothesis: Ichimoku TK cross on 6h with 1d trend filter (price above/below 1d Kumo cloud) to avoid counter-trend trades. 
The Kumo cloud acts as dynamic support/resistance. In bull markets (price > 1d Senkou Span A/B), we take long TK crosses; 
in bear markets (price < 1d Senkou Span A/B), we take short TK crosses. Volume confirmation (1.5x average) filters weak signals.
Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year). Works in bull/bear via 1d trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku and trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d OHLC for Ichimoku cloud calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Shift Senkou Spans forward by 26 periods (cloud is plotted ahead)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values are invalid due to shift
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_shifted)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_shifted)
    
    # === 6h OHLC for TK cross and price ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen and Kijun-sen on 6h for TK cross
    period_tenkan_6h = 9
    period_kijun_6h = 26
    max_high_tenkan_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    min_low_tenkan_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_sen_6h = (max_high_tenkan_6h + min_low_tenkan_6h) / 2
    
    max_high_kijun_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    min_low_kijun_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_sen_6h = (max_high_kijun_6h + min_low_kijun_6h) / 2
    
    # TK cross: Tenkan-sen crossing above/below Kijun-sen
    tk_cross_above = (tenkan_sen_6h > kijun_sen_6h) & (np.roll(tenkan_sen_6h, 1) <= np.roll(kijun_sen_6h, 1))
    tk_cross_below = (tenkan_sen_6h < kijun_sen_6h) & (np.roll(tenkan_sen_6h, 1) >= np.roll(kijun_sen_6h, 1))
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) 
            or np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])
            or np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        vol_avg = vol_ma[i]
        
        # Determine Kumo cloud boundaries (top and bottom of cloud)
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price above cloud (bullish 1d trend) + TK cross up + volume
            long_condition = (price > cloud_top) and tk_cross_above[i] and volume_confirmed
            # Short: price below cloud (bearish 1d trend) + TK cross down + volume
            short_condition = (price < cloud_bottom) and tk_cross_below[i] and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit if price falls below cloud bottom (trend invalidation)
            if price < cloud_bottom:
                signals[i] = 0.0
                position = 0
            # Exit on TK cross down (momentum loss)
            elif tk_cross_below[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price rises above cloud top (trend invalidation)
            if price > cloud_top:
                signals[i] = 0.0
                position = 0
            # Exit on TK cross up (momentum loss)
            elif tk_cross_above[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_v2"
timeframe = "6h"
leverage = 1.0