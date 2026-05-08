#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h data for Ichimoku conversion and base lines (using 6h period data)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 26:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1h).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1h, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1h, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1h, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1h, senkou_b)
    
    # 1d data for trend filter (Ichimoku cloud from daily)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Tenkan and Kijun
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # Daily Senkou Span A and B
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # Align daily cloud to 6h timeframe
    senkou_a_1d_6h = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_6h = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Volume filter: volume > 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 00-23 UTC (all hours for 6h)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_1d_6h[i]) or np.isnan(senkou_b_1d_6h[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom from daily Ichimoku
        cloud_top = max(senkou_a_1d_6h[i], senkou_b_1d_6h[i])
        cloud_bottom = min(senkou_a_1d_6h[i], senkou_b_1d_6h[i])
        
        if position == 0:
            # Long: TK cross bullish AND price above daily cloud
            tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
            price_above_cloud = close[i] > cloud_top
            volume_ok = volume[i] > vol_ma20[i]
            
            if tk_cross_bullish and price_above_cloud and volume_ok:
                signals[i] = 0.25
                position = 1
            
            # Short: TK cross bearish AND price below daily cloud
            tk_cross_bearish = tenkan_6h[i] < kijun_6h[i]
            price_below_cloud = close[i] < cloud_bottom
            
            if tk_cross_bearish and price_below_cloud and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross bearish OR price below daily cloud
            tk_cross_bearish = tenkan_6h[i] < kijun_6h[i]
            price_below_cloud = close[i] < cloud_bottom
            
            if tk_cross_bearish or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross bullish OR price above daily cloud
            tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
            price_above_cloud = close[i] > cloud_top
            
            if tk_cross_bullish or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals