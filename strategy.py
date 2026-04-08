#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d trend filter and volume confirmation
Hypothesis: Ichimoku TK cross above/below cloud on 6h, aligned with 1d Ichimoku trend (price above/below 1d cloud) and volume surge (current volume > 1.5x 20-period average) captures high-probability momentum. Works in bull/bear via cloud filter. Target: 15-30 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_trend_volume_v1"
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
    
    # 6h Ichimoku components (conversion=9, base=26, span=52)
    def ichimoku_components(high_arr, low_arr, close_arr):
        # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
        high_9 = pd.Series(high_arr).rolling(window=9, min_periods=9).max()
        low_9 = pd.Series(low_arr).rolling(window=9, min_periods=9).min()
        tenkan = (high_9 + low_9) / 2
        
        # Base Line (Kijun-sen): (26-period high + 26-period low)/2
        high_26 = pd.Series(high_arr).rolling(window=26, min_periods=26).max()
        low_26 = pd.Series(low_arr).rolling(window=26, min_periods=26).min()
        kijun = (high_26 + low_26) / 2
        
        # Leading Span A (Senkou Span A): (Conversion + Base)/2 shifted 26 ahead
        senkou_a = ((tenkan + kijun) / 2)
        
        # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2 shifted 26 ahead
        high_52 = pd.Series(high_arr).rolling(window=52, min_periods=52).max()
        low_52 = pd.Series(low_arr).rolling(window=52, min_periods=52).min()
        senkou_b = ((high_52 + low_52) / 2)
        
        # Lagging Span (Chikou Span): Close shifted 26 behind (not used for signals)
        return tenkan.values, kijun.values, senkou_a.values, senkou_b.values
    
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = ichimoku_components(high, low, close)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Ichimoku for trend filter (price relative to cloud)
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = ichimoku_components(high_1d, low_1d, close_1d)
    
    # Align 1d Ichimoku to 6h
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Need 52 periods for Senkou B
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_surge[i])):
            signals[i] = 0.0
            continue
        
        # Determine 6h cloud top and bottom
        cloud_top_6h = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom_6h = min(senkou_a_6h[i], senkou_b_6h[i])
        
        # Determine 1d cloud top and bottom
        cloud_top_1d = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom_1d = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        if position == 1:  # Long position
            # Exit: TK cross bearish OR price below 6h cloud OR price below 1d cloud
            tk_cross_bearish = tenkan_6h[i] < kijun_6h[i]
            price_below_6h_cloud = close[i] < cloud_bottom_6h
            price_below_1d_cloud = close[i] < cloud_bottom_1d
            if tk_cross_bearish or price_below_6h_cloud or price_below_1d_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross bullish OR price above 6h cloud OR price above 1d cloud
            tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
            price_above_6h_cloud = close[i] > cloud_top_6h
            price_above_1d_cloud = close[i] > cloud_top_1d
            if tk_cross_bullish or price_above_6h_cloud or price_above_1d_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # TK cross
            tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
            tk_cross_bearish = tenkan_6h[i] < kijun_6h[i]
            
            # Price relative to clouds
            price_above_6h_cloud = close[i] > cloud_top_6h
            price_below_6h_cloud = close[i] < cloud_bottom_6h
            price_above_1d_cloud = close[i] > cloud_top_1d
            price_below_1d_cloud = close[i] < cloud_bottom_1d
            
            # Long: TK cross bullish + price above both clouds + volume surge
            if (tk_cross_bullish and price_above_6h_cloud and price_above_1d_cloud and vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short: TK cross bearish + price below both clouds + volume surge
            elif (tk_cross_bearish and price_below_6h_cloud and price_below_1d_cloud and vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals