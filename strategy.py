#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Ichimoku Cloud filter + 6h Tenkan-Kijun cross.
# Uses 1d Ichimoku Cloud (Senkou Span A/B) to determine trend direction.
# Entry on 6h Tenkan-Kijun cross in direction of cloud color (green=bullish, red=bearish).
# Works in bull markets via bullish crosses in bullish cloud, and in bear markets via
# bearish crosses in bearish cloud. Avoids chop via cloud thickness filter.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_1d_Ichimoku_TK_Cross_CloudFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Ichimoku components on 1d timeframe
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2).shift(2)
    
    # Cloud color: green if Senkou A > Senkou B (bullish), red otherwise (bearish)
    cloud_green = senkou_a > senkou_b
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    cloud_green_aligned = align_htf_to_ltf(prices, df_1d, cloud_green.values.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure enough data for Ichimoku calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(cloud_green_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Previous values for crossover detection
        tenkan_prev = tenkan_aligned[i-1]
        kijun_prev = kijun_aligned[i-1]
        tenkan_curr = tenkan_aligned[i]
        kijun_curr = kijun_aligned[i]
        
        bullish_cross = tenkan_prev <= kijun_prev and tenkan_curr > kijun_curr
        bearish_cross = tenkan_prev >= kijun_prev and tenkan_curr < kijun_curr
        
        cloud_bullish = cloud_green_aligned[i] > 0.5
        cloud_bearish = cloud_green_aligned[i] <= 0.5
        
        if position == 0:
            # Long when bullish TK cross in bullish cloud
            if bullish_cross and cloud_bullish:
                signals[i] = 0.25
                position = 1
            # Short when bearish TK cross in bearish cloud
            elif bearish_cross and cloud_bearish:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on bearish TK cross or cloud turns bearish
            if bearish_cross or cloud_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on bullish TK cross or cloud turns bullish
            if bullish_cross or cloud_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals