#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d TK Cross filter and volume confirmation.
Long when price breaks above Ichimoku cloud (Senkou Span A/B) AND Tenkan > Kijun on 1d AND volume > 1.5x 20-period average.
Short when price breaks below Ichimoku cloud AND Tenkan < Kijun on 1d AND volume > 1.5x 20-period average.
Exit when price crosses the opposite cloud boundary (Tenkan-Kijun crossover on 6h).
Ichimoku provides dynamic support/resistance via cloud, TK cross filters for higher timeframe momentum,
volume confirmation reduces false breakouts. Designed to work in both bull and bear markets
by trading with 1d momentum while using cloud breaks for precise entry/exit.
Targets 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Get 1d data for TK Cross filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a_6h = ((tenkan_6h + kijun_6h) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b_6h = ((period52_high + period52_low) / 2)
    
    # Calculate TK Cross on 1d timeframe
    # Tenkan-sen 1d: (9-period high + 9-period low) / 2
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    # Kijun-sen 1d: (26-period high + 26-period low) / 2
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # Calculate volume average (20-period) on 6h
    volume_6h_series = pd.Series(volume_6h)
    volume_ma_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    tenkan_6h_aligned = align_htf_to_ltf(prices, df_6h, tenkan_6h)
    kijun_6h_aligned = align_htf_to_ltf(prices, df_6h, kijun_6h)
    senkou_a_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_a_6h)
    senkou_b_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_b_6h)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_6h_aligned[i]) or np.isnan(kijun_6h_aligned[i]) or 
            np.isnan(senkou_a_6h_aligned[i]) or np.isnan(senkou_b_6h_aligned[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        tenkan = tenkan_6h_aligned[i]
        kijun = kijun_6h_aligned[i]
        senkou_a = senkou_a_6h_aligned[i]
        senkou_b = senkou_b_6h_aligned[i]
        tenkan_1d_val = tenkan_1d_aligned[i]
        kijun_1d_val = kijun_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Cloud boundaries: Senkou Span A and B form the cloud
        upper_cloud = max(senkou_a, senkou_b)
        lower_cloud = min(senkou_a, senkou_b)
        
        if position == 0:
            # Long: price breaks above cloud AND Tenkan > Kijun on 1d (bullish momentum) AND volume > 1.5x avg
            if price > upper_cloud and tenkan_1d_val > kijun_1d_val and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud AND Tenkan < Kijun on 1d (bearish momentum) AND volume > 1.5x avg
            elif price < lower_cloud and tenkan_1d_val < kijun_1d_val and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below cloud OR Tenkan-Kijun cross down on 6h
            if price < lower_cloud or tenkan < kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above cloud OR Tenkan-Kijun cross up on 6h
            if price > upper_cloud or tenkan > kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTKCross_Volume"
timeframe = "6h"
leverage = 1.0