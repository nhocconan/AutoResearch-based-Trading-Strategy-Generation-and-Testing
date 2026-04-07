#!/usr/bin/env python3
"""
6h_adaptive_ichimoku_trend_follower
Hypothesis: On 6h timeframe, combine Ichimoku cloud (from daily) with Tenkan/Kijun crossover and price above/below cloud for trend direction. Add volume confirmation to filter false breakouts. The Ichimoku system provides dynamic support/resistance and trend identification, while the 6h timeframe reduces noise. Works in bull markets via cloud breakouts and in bear markets via cross-under of Tenkan/Kijun below cloud. Volume confirmation ensures institutional participation. Targets 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adaptive_ichimoku_trend_follower"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()
    tenkan = ((high_9 + low_9) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()
    kijun = ((high_26 + low_26) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): Current close shifted 26 periods back
    chikou = pd.Series(df_1d['close']).shift(26).values
    
    # Align Ichimoku components to 6h timeframe (shifted by 1 day for look-ahead prevention)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_6h = align_htf_to_ltf(prices, df_1d, chikou)
    
    # Current day's close for Chikou comparison (aligned)
    close_1d = pd.Series(df_1d['close']).values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume confirmation (24-period average on 6h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(chikou_6h[i]) or np.isnan(close_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 24-period average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if Tenkan crosses below Kijun (trend weakening)
            if tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]:
                exit_long = True
            # Exit if price closes below cloud bottom
            elif close[i] < cloud_bottom:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if Tenkan crosses above Kijun (trend weakening)
            if tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]:
                exit_short = True
            # Exit if price closes above cloud top
            elif close[i] > cloud_top:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Tenkan crosses above Kijun AND price above cloud AND Chikou above price from 26 periods ago AND volume confirmation
            long_entry = False
            if (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1] and
                close[i] > cloud_top and
                close_1d_aligned[i] > chikou_6h[i] and
                vol_confirm):
                long_entry = True
            
            # Short entry: Tenkan crosses below Kijun AND price below cloud AND Chikou below price from 26 periods ago AND volume confirmation
            short_entry = False
            if (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1] and
                close[i] < cloud_bottom and
                close_1d_aligned[i] < chikou_6h[i] and
                vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals