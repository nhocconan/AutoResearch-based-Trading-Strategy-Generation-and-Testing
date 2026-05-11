#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Trend_With_Volume"
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
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ichimoku components (10, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.8 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Ichimoku (52 periods)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + volume confirmation
            if (tenkan_aligned[i] > kijun_aligned[i] and  # TK cross bullish
                close[i] > cloud_top and                 # Price above cloud
                volume_filter[i]):                       # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish + price below cloud + volume confirmation
            elif (tenkan_aligned[i] < kijun_aligned[i] and  # TK cross bearish
                  close[i] < cloud_bottom and              # Price below cloud
                  volume_filter[i]):                       # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross bearish OR price below cloud
            if (tenkan_aligned[i] < kijun_aligned[i] or  # TK cross bearish
                close[i] < cloud_bottom):                # Price below cloud
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross bullish OR price above cloud
            if (tenkan_aligned[i] > kijun_aligned[i] or  # TK cross bullish
                close[i] > cloud_top):                   # Price above cloud
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals