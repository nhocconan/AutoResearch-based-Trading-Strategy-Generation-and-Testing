#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_HTF_Filter
Hypothesis: 6h Ichimoku TK cross with 12h cloud filter and volume confirmation. 
Ichimoku provides dynamic support/resistance via cloud and momentum via TK cross.
12h timeframe filters for higher-timeframe trend alignment to avoid counter-trend whipsaws.
Volume confirmation ensures breakouts have conviction. Discrete position sizing (0.25) controls risk.
Designed to work in both bull and bear markets by requiring alignment with 12h trend.
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
    
    # Get 12h data for cloud filter and trend
    df_12h = get_htf_data(prices, '12h')
    
    # 12h Ichimoku components (using standard periods: 9, 26, 52)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align 12h Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # 6h TK cross: Tenkan crosses Kijun
    # Calculate 6h Tenkan and Kijun for TK cross
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # TK cross signals: bullish when Tenkan > Kijun, bearish when Tenkan < Kijun
    tk_bullish = tenkan_6h > kijun_6h
    tk_bearish = tenkan_6h < kijun_6h
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Discrete position size
    
    # Warmup: need 52 periods for Senkou B, 26 for Kijun, 9 for Tenkan, 20 for volume
    start_idx = max(52, 26, 9, 20) + 5  # Extra buffer for alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tk_bullish[i]) or np.isnan(tk_bearish[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: TK cross with cloud filter and volume confirmation
            # Bullish: TK bullish cross AND price above cloud (uptrend)
            # Bearish: TK bearish cross AND price below cloud (downtrend)
            bullish_condition = (tk_bullish[i] and 
                               close_val > cloud_top_val and 
                               vol_conf)
            bearish_condition = (tk_bearish[i] and 
                                close_val < cloud_bottom_val and 
                                vol_conf)
            
            if bullish_condition:
                signals[i] = size
                position = 1
            elif bearish_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: TK bearish cross OR price drops below cloud bottom
            if (tk_bearish[i] or close_val < cloud_bottom_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TK bullish cross OR price rises above cloud top
            if (tk_bullish[i] or close_val > cloud_top_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_HTF_Filter"
timeframe = "6h"
leverage = 1.0