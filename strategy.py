#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with Daily Trend Filter
# - Tenkan-sen (9) / Kijun-sen (26) cross for entry signals on 6h
# - Senkou Span A/B cloud from 1d data as trend filter (price above/below cloud)
# - Only take long when price > 1d cloud, short when price < 1d cloud
# - Ichimoku provides clear support/resistance and momentum signals
# - Daily cloud filter ensures alignment with higher timeframe trend
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2
    
    # Shift Senkou spans forward by 26 periods
    senkou_span_a = np.roll(senkou_span_a, 26)
    senkou_span_b = np.roll(senkou_span_b, 26)
    senkou_span_a[:26] = np.nan
    senkou_span_b[:26] = np.nan
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 6h Tenkan-sen and Kijun-sen for crossover signals
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    high_9_6h = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    low_9_6h = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen_6h_calc = (high_9_6h + low_9_6h) / 2
    
    high_26_6h = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    low_26_6h = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen_6h_calc = (high_26_6h + low_26_6h) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or \
           np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or \
           np.isnan(tenkan_sen_6h_calc[i]) or np.isnan(kijun_sen_6h_calc[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine price position relative to 1d cloud
        cloud_top = max(senkou_span_a_6h[i], senkou_span_b_6h[i])
        cloud_bottom = min(senkou_span_a_6h[i], senkou_span_b_6h[i])
        price_above_cloud = close_6h[i] > cloud_top
        price_below_cloud = close_6h[i] < cloud_bottom
        
        # Determine 6h Tenkan/Kijun crossover
        tk_cross_up = tenkan_sen_6h_calc[i] > kijun_sen_6h_calc[i] and tenkan_sen_6h_calc[i-1] <= kijun_sen_6h_calc[i-1]
        tk_cross_down = tenkan_sen_6h_calc[i] < kijun_sen_6h_calc[i] and tenkan_sen_6h_calc[i-1] >= kijun_sen_6h_calc[i-1]
        
        if position == 0:
            # Long entry: TK cross up + price above 1d cloud
            if tk_cross_up and price_above_cloud:
                signals[i] = 0.25
                position = 1
            # Short entry: TK cross down + price below 1d cloud
            elif tk_cross_down and price_below_cloud:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross down or price falls below cloud
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross up or price rises above cloud
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_1dCloudFilter"
timeframe = "6h"
leverage = 1.0