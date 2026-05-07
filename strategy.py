#!/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_CloudFilter_12hTrend
# Hypothesis: Uses Ichimoku TK cross (Tenkan/Kijun) from 6h chart, filtered by cloud position from 12h trend and volume confirmation.
# Long: TK cross bullish + price above 12h cloud + volume spike.
# Short: TK cross bearish + price below 12h cloud + volume spike.
# Ichimoku cloud acts as dynamic support/resistance, reducing whipsaws in sideways markets.
# Target: 15-35 trades/year to minimize fee decay while capturing medium-term trends.

name = "6h_Ichimoku_TK_Cross_CloudFilter_12hTrend"
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
    
    # Get 12h data for Ichimoku calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Ichimoku parameters: Tenkan=9, Kijun=26, Senkou B=52
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Volume spike filter on 6h (24-period average ~ 4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud boundaries: Senkou A and Senkou B
        upper_cloud = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        lower_cloud = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 0:
            # Long: TK cross bullish (Tenkan > Kijun) + price above cloud + volume spike
            if tenkan_6h[i] > kijun_6h[i] and close[i] > upper_cloud and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish (Tenkan < Kijun) + price below cloud + volume spike
            elif tenkan_6h[i] < kijun_6h[i] and close[i] < lower_cloud and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross bearish OR price below cloud
            if tenkan_6h[i] < kijun_6h[i] or close[i] < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross bullish OR price above cloud
            if tenkan_6h[i] > kijun_6h[i] or close[i] > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals