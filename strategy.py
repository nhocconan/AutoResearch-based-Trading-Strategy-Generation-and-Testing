#!/usr/bin/env python3
"""
6h_1d_Ichimoku_Cloud_Breakout_Trend_Filter
Hypothesis: Ichimoku cloud on daily chart acts as major support/resistance. 
Price breaking above/below cloud with TK cross on 6h and volume confirmation captures trends in both bull and bear markets.
Cloud provides dynamic S/R, TK cross gives entry timing, trend filter avoids whipsaws.
Target: 12-37 trades/year per sensor.
"""

name = "6h_1d_Ichimoku_Cloud_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components on daily
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods back
    chikou = np.roll(close_1d, 26)
    chikou[:26] = np.nan
    
    # Align Ichimoku components to 6h (wait for daily close)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_6h = align_htf_to_ltf(prices, df_1d, chikou)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # TK Cross on 6h (using same Ichimoku but on 6h data for entry timing)
    # Calculate Ichimoku on 6h for TK cross only
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h_local = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h_local = (period26_high_6h + period26_low_6h) / 2
    
    tk_cross_above = (tenkan_6h_local > kijun_6h_local) & (np.roll(tenkan_6h_local, 1) <= np.roll(kijun_6h_local, 1))
    tk_cross_below = (tenkan_6h_local < kijun_6h_local) & (np.roll(tenkan_6h_local, 1) >= np.roll(kijun_6h_local, 1))
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # start after Ichimoku warmup
        # Get aligned values for current bar
        price = close[i]
        tk_above = tk_cross_above[i]
        tk_below = tk_cross_below[i]
        vol_conf = volume_conf[i]
        
        # Handle NaN values in cloud
        if np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: price above cloud, TK cross bullish, volume confirmation
            if price > cloud_top[i] and tk_above and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: price below cloud, TK cross bearish, volume confirmation
            elif price < cloud_bottom[i] and tk_below and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below cloud bottom or TK cross bearish
            if price < cloud_bottom[i] or tk_below:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above cloud top or TK cross bullish
            if price > cloud_top[i] or tk_above:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals