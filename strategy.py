#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6-hour Ichimoku Cloud with TK Cross + Cloud Filter from Daily
    # Uses Ichimoku (Tenkan/Kijun/Senkou) on 6h for entry timing and 1d cloud for trend filter
    # Tenkan-Kijun cross signals momentum shift, price relative to cloud filters trend direction
    # Works in both bull/bear markets by only taking trades in direction of higher timeframe trend
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Ichimoku components on 6h (Tenkan=9, Kijun=26, Senkou Span B=52)
    def ichimoku_components(high, low, close):
        # Tenkan-sen (Conversion Line): (9-period high + low)/2
        period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
        period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
        tenkan = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + low)/2
        period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
        period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
        kijun = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + low)/2
        period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
        period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
        senkou_b = (period52_high + period52_low) / 2
        
        return tenkan, kijun, senkou_a, senkou_b
    
    tenkan, kijun, senkou_a, senkou_b = ichimoku_components(high, low, close)
    
    # 1d Ichimoku cloud for trend filter (using same calculation)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Ichimoku components
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # Align 1d cloud to 6h
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Determine cloud boundaries (Senkou Span A and B form the cloud)
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top_1d = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom_1d = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top_1d[i]) or np.isnan(cloud_bottom_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK cross bullish (Tenkan > Kijun) + price above 1d cloud
            if tenkan[i] > kijun[i] and close[i] > cloud_top_1d[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish (Tenkan < Kijun) + price below 1d cloud
            elif tenkan[i] < kijun[i] and close[i] < cloud_bottom_1d[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: TK cross reverses or price enters the cloud
            if position == 1:
                if tenkan[i] < kijun[i] or close[i] < cloud_top_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if tenkan[i] > kijun[i] or close[i] > cloud_bottom_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dCloudFilter_v1"
timeframe = "6h"
leverage = 1.0