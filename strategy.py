#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Ichimoku cloud for trend direction, 6h price action for entry timing, and volume confirmation.
# Uses weekly Tenkan-sen/Kijun-sen cross as primary signal with cloud as filter.
# Enters long when Tenkan > Kijun AND price above cloud AND volume spike.
# Enters short when Tenkan < Kijun AND price below cloud AND volume spike.
# Exits when Tenkan/Kijun cross reverses.
# Designed for low trade frequency (15-30/year) to avoid fee drag. Ichimoku works in both trending and ranging markets.

name = "6h_1wIchimoku_TK_Cross_CloudFilter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Ichimoku calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components (weekly)
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    # Kijun-sen (Base Line): (26-period high + low)/2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    # Senkou Span B (Leading Span B): (52-period high + low)/2
    
    # Tenkan-sen (9-period)
    high_9 = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (26-period)
    high_26 = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span B (52-period)
    high_52 = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2.0
    
    # Senkou Span A = (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    # Volume confirmation: 6h volume spike (2x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure enough data for Senkou B
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Enter long: Tenkan > Kijun AND price above cloud AND volume spike
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                close[i] > cloud_top and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Tenkan < Kijun AND price below cloud AND volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  close[i] < cloud_bottom and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan/Kijun cross reverses (Tenkan < Kijun)
            if tenkan_aligned[i] < kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan/Kijun cross reverses (Tenkan > Kijun)
            if tenkan_aligned[i] > kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals