#!/usr/bin/env python3
# 6h_ichimoku_kumo_breakout_1d_trend
# Hypothesis: Ichimoku cloud breakout on 6h with 1d Kumo (cloud) trend filter and volume confirmation.
# Long when price breaks above 6h Tenkan/Kijun cross AND price above 1d Kumo (cloud top) with volume > 1.5x avg.
# Short when price breaks below 6h Tenkan/Kijun cross AND price below 1d Kumo (cloud bottom) with volume > 1.5x avg.
# Exit when Tenkan/Kijun cross reverses or price crosses Kumo mid-line on 6h.
# Ichimoku provides dynamic support/resistance; Kumo filter ensures trend alignment across timeframes.
# Target: 60-120 total trades over 4 years (~15-30/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_kumo_breakout_1d_trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Kumo (cloud) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    tenkan_1d = (high_1d_series.rolling(window=9, min_periods=9).max() + 
                 low_1d_series.rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_1d = (high_1d_series.rolling(window=26, min_periods=26).max() + 
                low_1d_series.rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b_1d = ((high_1d_series.rolling(window=52, min_periods=52).max() + 
                    low_1d_series.rolling(window=52, min_periods=52).min()) / 2).shift(26)
    # Kumo (cloud) top/bottom: Senkou Span A and B
    kumotop_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumobottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 1d Ichimoku components to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    kumotop_1d_aligned = align_htf_to_ltf(prices, df_1d, kumotop_1d.values)
    kumobottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumobottom_1d.values)
    
    # Calculate Ichimoku on 6h for entry signals
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    # Tenkan-sen (9-period)
    tenkan_6h = (high_series.rolling(window=9, min_periods=9).max() + 
                 low_series.rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (26-period)
    kijun_6h = (high_series.rolling(window=26, min_periods=26).max() + 
                low_series.rolling(window=26, min_periods=26).min()) / 2
    # Kumo (cloud) on 6h for exit
    senkou_a_6h = ((tenkan_6h + kijun_6h) / 2).shift(26)
    senkou_b_6h = ((high_series.rolling(window=52, min_periods=52).max() + 
                    low_series.rolling(window=52, min_periods=52).min()) / 2).shift(26)
    kumotop_6h = np.maximum(senkou_a_6h, senkou_b_6h)
    kumobottom_6h = np.minimum(senkou_a_6h, senkou_b_6h)
    kumomid_6h = (kumotop_6h + kumobottom_6h) / 2
    
    # Volume confirmation (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup (need 52 for Ichimoku calculations)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(kumotop_1d_aligned[i]) or np.isnan(kumobottom_1d_aligned[i]) or
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Tenkan/Kijun cross turns bearish OR price below Kumo mid
            tenkan_kijun_cross = tenkan_6h[i] < kijun_6h[i]
            price_below_kumo = close[i] < kumomid_6h[i]
            if tenkan_kijun_cross or price_below_kumo:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan/Kijun cross turns bullish OR price above Kumo mid
            tenkan_kijun_cross = tenkan_6h[i] > kijun_6h[i]
            price_above_kumo = close[i] > kumomid_6h[i]
            if tenkan_kijun_cross or price_above_kumo:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Tenkan/Kijun cross on 6h
            tenkan_kijun_cross_up = tenkan_6h[i] > kijun_6h[i]
            tenkan_kijun_cross_down = tenkan_6h[i] < kijun_6h[i]
            
            # Price relative to 1d Kumo (cloud)
            price_above_kumo = close[i] > kumotop_1d_aligned[i]
            price_below_kumo = close[i] < kumobottom_1d_aligned[i]
            
            # Entry conditions
            if (tenkan_kijun_cross_up and price_above_kumo and volume_ok):
                position = 1
                signals[i] = 0.25
            elif (tenkan_kijun_cross_down and price_below_kumo and volume_ok):
                position = -1
                signals[i] = -0.25
    
    return signals