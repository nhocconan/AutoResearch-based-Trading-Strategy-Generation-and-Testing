# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_Trend
Hypothesis: Uses Ichimoku cloud (Tenkan-sen, Kijun-sen, Senkou Span A/B) on the daily timeframe to identify trend direction and momentum. Enters long when price breaks above the cloud with bullish TK cross in uptrend, short when price breaks below the cloud with bearish TK cross in downtrend. Uses volume spike confirmation to filter false breakouts. Designed to capture trend continuation moves in both bull and bear markets by trading with the higher timeframe Ichimoku trend. Targets 50-150 total trades over 4 years to minimize fee drag while capturing meaningful momentum.
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
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    # Align Ichimoku components to 6h timeframe (wait for daily close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Calculate cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Determine trend: price above cloud = uptrend, below cloud = downtrend
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # TK Cross signals
    tk_cross_bullish = tenkan_sen_aligned > kijun_sen_aligned
    tk_cross_bearish = tenkan_sen_aligned < kijun_sen_aligned
    
    # Volume spike confirmation (>1.8x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for Ichimoku to stabilize (52 + 26 shift)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (vol_spike[i] and 
                     price_above_cloud[i] and 
                     tk_cross_bullish[i] and 
                     close[i] > close[i-1])  # Confirm with upward momentum
        
        short_entry = (vol_spike[i] and 
                      price_below_cloud[i] and 
                      tk_cross_bearish[i] and 
                      close[i] < close[i-1])  # Confirm with downward momentum
        
        # Exit conditions: TK cross reversal or price returns to cloud
        long_exit = (not tk_cross_bullish[i]) or (close[i] < cloud_top[i])
        short_exit = (not tk_cross_bearish[i]) or (close[i] > cloud_bottom[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_Trend"
timeframe = "6h"
leverage = 1.0