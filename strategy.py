#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d Trend Filter and Volume Spike
# Uses Tenkan/Kijun cross + price above/below cloud (from 1d) + volume confirmation.
# Works in bull markets (long above cloud) and bear markets (short below cloud).
# The cloud acts as dynamic support/resistance, reducing whipsaw.
# Target: 50-150 total trades over 4 years = 12-37/year.
# Timeframe: 6h, HTF: 1d

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou B
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ichimoku calculations (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma[i])):
            continue
        
        # Long entry: Tenkan > Kijun, price above cloud, volume spike
        if (tenkan_6h[i] > kijun_6h[i] and
            close[i] > cloud_top[i] and
            volume[i] > 2.0 * vol_ma[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Tenkan < Kijun, price below cloud, volume spike
        elif (tenkan_6h[i] < kijun_6h[i] and
              close[i] < cloud_bottom[i] and
              volume[i] > 2.0 * vol_ma[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Tenkan/Kijun cross in opposite direction or price enters cloud
        elif position == 1 and (tenkan_6h[i] < kijun_6h[i] or close[i] < cloud_top[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tenkan_6h[i] > kijun_6h[i] or close[i] > cloud_bottom[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_Cloud_Volume"
timeframe = "6h"
leverage = 1.0