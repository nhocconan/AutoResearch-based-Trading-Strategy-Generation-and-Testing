#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_12hTrend_VolumeSpike
Hypothesis: Uses Ichimoku cloud from 1d timeframe for trend filter, with TK cross on 6h for entry timing.
Enter long when price breaks above 1d cloud AND TK cross bullish on 6h AND volume > 1.8 * 20-period average.
Enter short when price breaks below 1d cloud AND TK cross bearish on 6h AND volume > 1.8 * 20-period average.
Exit when price returns to opposite cloud boundary OR TK cross reverses.
Ichimoku cloud provides strong support/resistance in both bull and bear markets, TK cross gives timely entries.
High volume threshold filters weak breakouts. Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position size.
Designed to work in ranging markets (cloud acts as dynamic S/R) and trending markets (breakouts).
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
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components on 1d (using standard periods: 9, 26, 52)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    # Note: Senkou spans are already plotted ahead, so we need to align properly
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud boundaries: upper cloud = max(Senkou A, Senkou B), lower cloud = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a_aligned, senkou_b_aligned)
    lower_cloud = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # TK cross on 6h timeframe for entry timing
    # Tenkan-sen and Kijun-sen on 6h
    period_tenkan_6h = 9
    period_kijun_6h = 26
    high_tenkan_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    low_tenkan_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_6h = (high_tenkan_6h + low_tenkan_6h) / 2.0
    
    high_kijun_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    low_kijun_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_6h = (high_kijun_6h + low_kijun_6h) / 2.0
    
    # TK cross signals: bullish when Tenkan > Kijun, bearish when Tenkan < Kijun
    tk_bullish = tenkan_6h > kijun_6h
    tk_bearish = tenkan_6h < kijun_6h
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d Ichimoku (52), 6h TK cross (26), volume avg (20)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or 
            np.isnan(tk_bullish[i]) or np.isnan(tk_bearish[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_cloud_val = upper_cloud[i]
        lower_cloud_val = lower_cloud[i]
        tk_bull = tk_bullish[i]
        tk_bear = tk_bearish[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: price breaks cloud with TK cross alignment AND volume
            # Long: price breaks above upper cloud AND TK bullish AND volume
            long_condition = (close_val > upper_cloud_val) and tk_bull and vol_conf
            # Short: price breaks below lower cloud AND TK bearish AND volume
            short_condition = (close_val < lower_cloud_val) and tk_bear and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to lower cloud OR TK cross turns bearish
            exit_condition = (close_val <= lower_cloud_val) or (not tk_bull)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to upper cloud OR TK cross turns bullish
            exit_condition = (close_val >= upper_cloud_val) or (not tk_bear)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0