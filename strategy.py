#!/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
# Hypothesis: 6h strategy using Ichimoku Tenkan/Kijun cross with cloud filter from 1d timeframe.
# Enters long when Tenkan crosses above Kijun on 6h, price above 1d cloud (bullish), and price > 1d Senkou Span A.
# Enters short when Tenkan crosses below Kijun on 6h, price below 1d cloud (bearish), and price < 1d Senkou Span B.
# Uses 1d Ichimoku cloud to filter trades in direction of higher timeframe trend, avoiding counter-trend whipsaws.
# Designed for low trade frequency (12-37/year) with strong trend filtration to work in both bull and bear markets.

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1d = (high_52 + low_52) / 2
    
    # Align all 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h Tenkan and Kijun for crossover signal
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (high_9_6h + low_9_6h) / 2
    
    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (high_26_6h + low_26_6h) / 2
    
    # Calculate crossover signals: 1 for bullish cross (Tenkan > Kijun), -1 for bearish cross (Tenkan < Kijun)
    tk_cross = np.zeros(n)
    tk_cross[1:] = np.where(tenkan_6h[1:] > kijun_6h[1:], 1,
                           np.where(tenkan_6h[1:] < kijun_6h[1:], -1, 0))
    # Only act on change in crossover state
    tk_cross_change = np.diff(tk_cross, prepend=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 9)  # Need 26-period data for Kijun
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud status and trend from 1d Ichimoku
        # Bullish: price above Senkou Span A, Senkou A > Senkou B
        # Bearish: price below Senkou Span B, Senkou A < Senkou B
        bullish_cloud = (close[i] > senkou_a_1d_aligned[i]) and (senkou_a_1d_aligned[i] > senkou_b_1d_aligned[i])
        bearish_cloud = (close[i] < senkou_b_1d_aligned[i]) and (senkou_a_1d_aligned[i] < senkou_b_1d_aligned[i])
        
        if position == 0:
            # Long: bullish TK cross on 6h, bullish cloud on 1d
            if (tk_cross_change[i] == 2 and bullish_cloud):  # Change from -1 to 1 = bullish cross
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross on 6h, bearish cloud on 1d
            elif (tk_cross_change[i] == -2 and bearish_cloud):  # Change from 1 to -1 = bearish cross
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross turns bearish OR price enters cloud (Senkou A < price < Senkou B)
            if (tk_cross_change[i] == -2) or (senkou_b_1d_aligned[i] <= close[i] <= senkou_a_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross turns bullish OR price enters cloud
            if (tk_cross_change[i] == 2) or (senkou_b_1d_aligned[i] <= close[i] <= senkou_a_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals