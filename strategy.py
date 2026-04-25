#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_TK_Cross_1dTrend
Hypothesis: Use 1d Ichimoku cloud (Senkou Span A/B) as trend filter and TK cross on 6h for entry.
In bull markets: price above cloud + TK cross up = long.
In bear markets: price below cloud + TK cross down = short.
Adds volume confirmation to reduce false signals.
Designed for 6h timeframe to target 12-30 trades/year (50-120 over 4 years).
Uses discrete position sizing (0.25) to manage drawdown in 2022-like crashes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe (completed 1d cloud only)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)
    
    # Calculate TK cross on 6h (using actual 6h high/low)
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h_actual = (high_9_6h + low_9_6h) / 2
    
    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h_actual = (high_26_6h + low_26_6h) / 2
    
    # Volume confirmation: current volume > 2.0 * average of last 20 periods
    if n >= 20:
        vol_avg = np.zeros(n)
        vol_avg[20:] = np.convolve(volume, np.ones(20)/20, mode='valid')
        vol_avg[:20] = vol_avg[20] if n > 20 else 0
        volume_spike = volume > 2.0 * vol_avg
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52) and TK cross (26)
    start_idx = max(52 + 26, 26)  # 52 for Senkou B, +26 for shift, 26 for TK cross
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(tenkan_6h_actual[i]) or np.isnan(kijun_6h_actual[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 0:
            # Long: price above cloud AND TK cross up (tenkan > kijun) AND volume spike
            long_setup = (close[i] > cloud_top) and \
                         (tenkan_6h_actual[i] > kijun_6h_actual[i]) and \
                         volume_spike[i]
            # Short: price below cloud AND TK cross down (tenkan < kijun) AND volume spike
            short_setup = (close[i] < cloud_bottom) and \
                          (tenkan_6h_actual[i] < kijun_6h_actual[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters cloud OR TK cross down
            if (close[i] < cloud_top) or (tenkan_6h_actual[i] < kijun_6h_actual[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters cloud OR TK cross up
            if (close[i] > cloud_bottom) or (tenkan_6h_actual[i] > kijun_6h_actual[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_TK_Cross_1dTrend"
timeframe = "6h"
leverage = 1.0