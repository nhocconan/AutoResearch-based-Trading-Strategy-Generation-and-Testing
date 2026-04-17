#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high9 + low9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high26 + low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high52 + low52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need Ichimoku components
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Ichimoku signals
        tk_cross_bull = tenkan_6h[i] > kijun_6h[i]  # Tenkan above Kijun
        tk_cross_bear = tenkan_6h[i] < kijun_6h[i]  # Tenkan below Kijun
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        if position == 0:
            # Long: TK cross bullish AND price above cloud with volume
            if (tk_cross_bull and price_above_cloud and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish AND price below cloud with volume
            elif (tk_cross_bear and price_below_cloud and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross bearish OR price drops below cloud
            if (tk_cross_bear) or (close[i] < cloud_top[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above cloud
            if (tk_cross_bull) or (close[i] > cloud_bottom[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Volume"
timeframe = "6h"
leverage = 1.0