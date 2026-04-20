#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Ichimoku_CloudBreak_VolumeTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # === Daily Ichimoku Components ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high9 + low9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    high26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high26 + low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    high52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high52 + low52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_a = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_a = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_a = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_a = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # === Volume Trend Filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        tenkan_val = tenkan_a[i]
        kijun_val = kijun_a[i]
        senkou_a_val = senkou_a_a[i]
        senkou_b_val = senkou_b_a[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(tenkan_val) or 
            np.isnan(kijun_val) or np.isnan(senkou_a_val) or 
            np.isnan(senkou_b_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Long: Price above cloud, Tenkan > Kijun (bullish), volume confirmation
            if close_val > cloud_top and tenkan_val > kijun_val and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, Tenkan < Kijun (bearish), volume confirmation
            elif close_val < cloud_bottom and tenkan_val < kijun_val and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls below cloud OR Tenkan < Kijun
            if close_val < cloud_top or tenkan_val < kijun_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above cloud OR Tenkan > Kijun
            if close_val > cloud_bottom or tenkan_val > kijun_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals