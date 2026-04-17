#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Ichimoku cloud (Senkou Span A/B) and TK cross (Tenkan/Kijun) for trend direction, 
combined with 1w volume spike confirmation. Long when price is above cloud, TK cross bullish, and 1w volume > 2x 20-period average. 
Short when price is below cloud, TK cross bearish, and 1w volume > 2x 20-period average. 
Ichimoku provides dynamic support/resistance and trend identification; volume spike confirms institutional participation. 
Works in bull (trend following) and bear (counter-trend at extremes via cloud breaks). 
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
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
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
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
    
    # Get 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w volume 20-period average
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 52  # need enough for Senkou B
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ma_20_1w_aligned[i]) or np.isnan(volume_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 2x 20-period average
        volume_confirmed = volume_1w_aligned[i] > 2.0 * vol_ma_20_1w_aligned[i]
        
        # Determine cloud top and bottom (Senkou Span A and B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK cross: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        if position == 0:
            # Long: price above cloud, TK cross bullish, volume confirmed
            if (close[i] > cloud_top and 
                tk_bullish and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, TK cross bearish, volume confirmed
            elif (close[i] < cloud_bottom and 
                  tk_bearish and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below cloud or TK cross turns bearish
            if (close[i] < cloud_top or 
                not tk_bullish):  # Tenkan <= Kijun
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above cloud or TK cross turns bullish
            if (close[i] > cloud_bottom or 
                not tk_bearish):  # Tenkan >= Kijun
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dIchimoku_1wVolume"
timeframe = "6h"
leverage = 1.0