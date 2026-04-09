#!/usr/bin/env python3
# 6h_ichimoku_cloud_breakout_v1
# Hypothesis: 6h strategy using Ichimoku cloud from 1d timeframe for trend direction,
# with TK cross on 6h for entry timing and volume confirmation.
# Long: Price above 1d cloud, TK cross bullish on 6h, volume > 1.5x 20-period average
# Short: Price below 1d cloud, TK cross bearish on 6h, volume > 1.5x 20-period average
# Exit: TK cross in opposite direction or price crosses cloud midpoint (Kijun-sen)
# Uses 6h primary timeframe with 1d HTF for Ichimoku cloud and 6h for TK cross and volume.
# Ichimoku components: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (52-period displaced)
# Works in both bull and bear markets by following the higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components for 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 displaced 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 displaced 26 periods ahead
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_52 + low_52) / 2.0
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h TK cross (Tenkan-sen/Kijun-sen crossover)
    # Tenkan-sen (9-period) on 6h
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (high_9_6h + low_9_6h) / 2.0
    
    # Kijun-sen (26-period) on 6h
    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (high_26_6h + low_26_6h) / 2.0
    
    # TK cross signals: 1 = bullish cross (Tenkan > Kijun), -1 = bearish cross (Tenkan < Kijun), 0 = no cross
    tk_cross = np.zeros(n)
    tk_cross[1:] = np.where(
        (tenkan_6h[1:] > kijun_6h[1:]) & (tenkan_6h[:-1] <= kijun_6h[:-1]), 1,
        np.where(
            (tenkan_6h[1:] < kijun_6h[1:]) & (tenkan_6h[:-1] >= kijun_6h[:-1]), -1, 0
        )
    )
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup period for all indicators
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries and midpoint
        cloud_top = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_midpoint = (senkou_a_1d_aligned[i] + senkou_b_1d_aligned[i]) / 2.0
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: TK cross bearish or price crosses below cloud midpoint
            if tk_cross[i] == -1 or close[i] < cloud_midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross bullish or price crosses above cloud midpoint
            if tk_cross[i] == 1 or close[i] > cloud_midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price above cloud, TK cross bullish, volume confirmed
            if (close[i] > cloud_top and 
                tk_cross[i] == 1 and 
                volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below cloud, TK cross bearish, volume confirmed
            elif (close[i] < cloud_bottom and 
                  tk_cross[i] == -1 and 
                  volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals