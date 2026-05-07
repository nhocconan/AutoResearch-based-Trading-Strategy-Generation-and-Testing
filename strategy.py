#!/usr/bin/env python3
"""
6H_Ichimoku_Cloud_Filter_1D_Trend_v1
Hypothesis: Use Ichimoku cloud from daily timeframe as trend filter and TK cross on 6h for entry.
Long when price is above daily cloud and TK line crosses above Kijun on 6h.
Short when price is below daily cloud and TK line crosses below Kijun on 6h.
Volume confirmation: current volume > 1.5x 20-period average volume.
Ichimoku cloud provides strong trend identification and support/resistance,
while TK cross gives timely entries. Works in both bull (cloud as support) and bear (cloud as resistance).
"""
name = "6H_Ichimoku_Cloud_Filter_1D_Trend_v1"
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
    
    # Get daily data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted -22 periods (not used for trend)
    
    # Cloud top and bottom (Senkou Span A and B)
    # Shift forward by 26 periods for cloud
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values are invalid due to shift
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Align Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_shifted)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_shifted)
    
    # Get 6h data for TK cross
    tenkan_6h = pd.Series(close).rolling(window=9, min_periods=9).apply(
        lambda x: (x.max() + x.min()) / 2, raw=True).values
    kijun_6h = pd.Series(close).rolling(window=26, min_periods=26).apply(
        lambda x: (x.max() + x.min()) / 2, raw=True).values
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(52, 26, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        # Determine cloud color and price position relative to cloud
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            # Minimum 6 bars between trades (1.5 days on 6h TF) to reduce frequency
            if bars_since_exit < 6:
                continue
                
            # TK cross signals
            tk_cross_above = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
            tk_cross_below = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
            
            # Long: price above cloud and TK crosses above Kijun
            if price_above_cloud and tk_cross_above and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price below cloud and TK crosses below Kijun
            elif price_below_cloud and tk_cross_below and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: TK cross in opposite direction or price re-enters cloud
            tk_cross_above = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
            tk_cross_below = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
            
            if position == 1:
                # Exit long: TK cross below or price re-enters cloud from above
                if tk_cross_below or (close[i] <= cloud_top and close[i] >= cloud_bottom):
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: TK cross above or price re-enters cloud from below
                if tk_cross_above or (close[i] >= cloud_bottom and close[i] <= cloud_top):
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = -0.25
    
    return signals