#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with price above/below 1d cloud for trend alignment.
Ichimoku provides built-in trend, momentum, and support/resistance. Using 1d cloud (Senkou Span A/B) as higher-timeframe trend filter reduces whipsaws in both bull and bear markets.
Tenkan-Kijun cross gives timely entries while cloud filter ensures we only trade with the 1d trend. Discrete position sizing (0.25) minimizes fee churn.
Target: 12-30 trades/year per symbol for low fee drag and strong test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for cloud and trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # === 6h OHLC for Ichimoku calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for signals)
    
    # === 1d Ichimoku cloud for trend filter ===
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    # 1d Tenkan-sen (9-period)
    df_1d_period9_high = pd.Series(df_1d_high).rolling(window=9, min_periods=9).max().values
    df_1d_period9_low = pd.Series(df_1d_low).rolling(window=9, min_periods=9).min().values
    df_1d_tenkan = (df_1d_period9_high + df_1d_period9_low) / 2
    
    # 1d Kijun-sen (26-period)
    df_1d_period26_high = pd.Series(df_1d_high).rolling(window=26, min_periods=26).max().values
    df_1d_period26_low = pd.Series(df_1d_low).rolling(window=26, min_periods=26).min().values
    df_1d_kijun = (df_1d_period26_high + df_1d_period26_low) / 2
    
    # 1d Senkou Span A
    df_1d_senkou_a = ((df_1d_tenkan + df_1d_kijun) / 2)
    
    # 1d Senkou Span B (52-period)
    df_1d_period52_high = pd.Series(df_1d_high).rolling(window=52, min_periods=52).max().values
    df_1d_period52_low = pd.Series(df_1d_low).rolling(window=52, min_periods=52).min().values
    df_1d_senkou_b = ((df_1d_period52_high + df_1d_period52_low) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe
    # Senkou Span A and B need to be shifted forward by 26 periods after alignment
    df_1d_senkou_a_aligned = align_htf_to_ltf(prices, df_1d, df_1d_senkou_a)
    df_1d_senkou_b_aligned = align_htf_to_ltf(prices, df_1d, df_1d_senkou_b)
    
    # The cloud is formed by Senkou Span A and B
    # For trend filter: price above cloud = bullish, below cloud = bearish
    # We need to shift the cloud forward by 26 periods (as per Ichimoku rules)
    # Since align_htf_to_ltf already handles the HTF bar completion delay,
    # we add the Ichomoku cloud shift (26 periods) as additional delay
    cloud_top = np.maximum(df_1d_senkou_a_aligned, df_1d_senkou_b_aligned)
    cloud_bottom = np.minimum(df_1d_senkou_a_aligned, df_1d_senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Warmup for Ichimoku calculations
        # Skip if indicators not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Ichimoku signals:
            # Tenkan-sen crossing above Kijun-sen = bullish signal
            # Tenkan-sen crossing below Kijun-sen = bearish signal
            tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            
            # Cloud filter: price above cloud = bullish bias, below cloud = bearish bias
            price_above_cloud = price > cloud_top[i]
            price_below_cloud = price < cloud_bottom[i]
            
            # Entry logic: TK cross in direction of cloud filter
            if tk_cross_up and price_above_cloud:
                signals[i] = 0.25
                position = 1
            elif tk_cross_down and price_below_cloud:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Tenkan-sen crosses below Kijun-sen OR price drops below cloud
            tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            price_below_cloud = price < cloud_bottom[i]
            
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Tenkan-sen crosses above Kijun-sen OR price rises above cloud
            tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            price_above_cloud = price > cloud_top[i]
            
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0