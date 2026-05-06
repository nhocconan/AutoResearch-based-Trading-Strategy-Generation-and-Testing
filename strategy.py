#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku Cloud with Tenkan/Kijun cross and cloud color filter
# - Uses 1d Ichimoku system (Tenkan, Kijun, Senkou A/B, Chikou) for trend direction
# - Enters long when Tenkan crosses above Kijun AND price is above cloud (bullish)
# - Enters short when Tenkan crosses below Kijun AND price is below cloud (bearish)
# - Exits when Tenkan/Kijun cross reverses OR price crosses into opposite cloud
# - Ichimoku provides multi-factor trend confirmation suitable for 6h timeframe
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_1dIchimoku_TK_Cross_Cloud"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    n1 = 9   # Tenkan-sen period
    n2 = 26  # Kijun-sen period
    n3 = 52  # Senkou Span B period
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=n1, min_periods=n1).max() + 
              pd.Series(low).rolling(window=n1, min_periods=n1).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=n2, min_periods=n2).max() + 
             pd.Series(low).rolling(window=n2, min_periods=n2).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=n3, min_periods=n3).max() + 
                 pd.Series(low).rolling(window=n3, min_periods=n3).min()) / 2)
    
    # Chikou Span (Lagging Span): Close shifted -26 periods
    chikou = pd.Series(close).shift(-n2)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_6h = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # Cloud top and bottom (Senkou Span A/B)
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Tenkan/Kijun cross signals
    tk_cross_up = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_down = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any critical value is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud
            if tk_cross_up[i] and price_above_cloud[i]:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below cloud
            elif tk_cross_down[i] and price_below_cloud[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan/Kijun cross down OR price crosses below cloud
            if tk_cross_down[i] or close[i] < cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan/Kijun cross up OR price crosses above cloud
            if tk_cross_up[i] or close[i] > cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals