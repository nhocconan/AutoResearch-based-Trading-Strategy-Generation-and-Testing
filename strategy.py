#!/usr/bin/env python3
# 6h_1d_Ichimoku_Cloud_Filter
# Hypothesis: Use 1d Ichimoku Cloud as trend filter on 6h timeframe with TK cross entry.
# The Ichimoku Cloud from daily timeframe provides institutional-grade support/resistance.
# TK (Tenkan-Kijun) cross on 6h provides timely entries in direction of higher timeframe trend.
# Works in bull markets (trend following with cloud support) and bear markets (avoiding false breaks).
# Cloud acts as dynamic support/resistance reducing whipsaws. Targets 12-37 trades/year.

name = "6h_1d_Ichimoku_Cloud_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate TK cross on 6h
    high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h_internal = (high_6h + low_6h) / 2
    
    high_kijun_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_kijun_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h_internal = (high_kijun_6h + low_kijun_6h) / 2
    
    tk_cross = tenkan_6h_internal - kijun_6h_internal
    tk_cross_prev = np.roll(tk_cross, 1)
    tk_cross_prev[0] = 0
    
    # Determine cloud (green/red) and price position
    # Green cloud: Senkou A > Senkou B (bullish)
    # Red cloud: Senkou A < Senkou B (bearish)
    cloud_green = senkou_a_6h > senkou_b_6h
    
    # Price above/below cloud
    price_above_cloud = (close > senkou_a_6h) & (close > senkou_b_6h)
    price_below_cloud = (close < senkou_a_6h) & (close < senkou_b_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(tk_cross[i]) or np.isnan(tk_cross_prev[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TK cross bullish + price above cloud (bullish alignment)
            if (tk_cross[i] > 0 and tk_cross_prev[i] <= 0 and 
                price_above_cloud[i] and cloud_green[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish + price below cloud (bearish alignment)
            elif (tk_cross[i] < 0 and tk_cross_prev[i] >= 0 and 
                  price_below_cloud[i] and not cloud_green[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross bearish OR price breaks below cloud
            if (tk_cross[i] < 0 and tk_cross_prev[i] >= 0) or price_below_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross bullish OR price breaks above cloud
            if (tk_cross[i] > 0 and tk_cross_prev[i] <= 0) or price_above_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals