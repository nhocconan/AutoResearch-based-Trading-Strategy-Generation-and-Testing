#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Ichimoku_Kumo_Twist_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 4h
    tenkan_a = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_a = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_a = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_a = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Kumo (Cloud) top and bottom
    kumo_top = np.maximum(senkou_a_a, senkou_b_a)
    kumo_bottom = np.minimum(senkou_a_a, senkou_b_a)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_a[i]) or np.isnan(kijun_a[i]) or np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Kumo twist: Senkou A crossing above/below Senkou B
        kumo_twist_up = senkou_a_a[i] > senkou_b_a[i] and senkou_a_a[i-1] <= senkou_b_a[i-1]
        kumo_twist_down = senkou_a_a[i] < senkou_b_a[i] and senkou_a_a[i-1] >= senkou_b_a[i-1]
        
        if position == 0:
            # Long: price above cloud, Tenkan > Kijun, Kumo twisting up, volume
            if price > kumo_top[i] and tenkan_a[i] > kijun_a[i] and kumo_twist_up and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, Tenkan < Kijun, Kumo twisting down, volume
            elif price < kumo_bottom[i] and tenkan_a[i] < kijun_a[i] and kumo_twist_down and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price drops below cloud bottom or Tenkan < Kijun
            if price < kumo_bottom[i] or tenkan_a[i] < kijun_a[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price rises above cloud top or Tenkan > Kijun
            if price > kumo_top[i] or tenkan_a[i] > kijun_a[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals