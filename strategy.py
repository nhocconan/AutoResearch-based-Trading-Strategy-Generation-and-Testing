#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter_VolumeConfirm
Hypothesis: On 6h timeframe, Ichimoku Tenkan-Kijun (TK) cross signals are filtered by 1d Ichimoku cloud (price above/below cloud) and volume confirmation (>1.5x 20-bar average). TK cross provides timely momentum signals, while 1d cloud filter ensures alignment with higher timeframe trend, reducing false signals in ranging markets. Volume confirmation ensures institutional participation. Designed for low trade frequency (15-25/year) to minimize fee drag in 6h timeframe. Works in both bull and bear markets via cloud filter (price above cloud = bullish bias, below = bearish bias).
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
    
    # Get 1d data for HTF Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for filtering)
    
    # The Ichimoku cloud is between Senkou Span A and Senkou Span B
    # We need the cloud values from 26 periods ago to align with current price
    # Shift Senkou spans back by 26 to get current cloud
    senkou_a_current = np.concatenate([np.full(26, np.nan), senkou_a_1d[:-26]])
    senkou_b_current = np.concatenate([np.full(26, np.nan), senkou_b_1d[:-26]])
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_current)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_current)
    
    # Calculate 6h Ichimoku for TK cross
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(50, 26)  # Ichimoku needs 26+ periods, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        tenkan_1d_val = tenkan_aligned[i]
        kijun_1d_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        tenkan_6h_val = tenkan_6h[i]
        kijun_6h_val = kijun_6h[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Determine cloud boundaries (top and bottom of cloud)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol_val > 1.5 * vol_ma_val
        
        # TK cross conditions
        tk_cross_up = (tenkan_6h_val > kijun_6h_val) and (tenkan_6h[i-1] <= kijun_6h[i-1])
        tk_cross_down = (tenkan_6h_val < kijun_6h_val) and (tenkan_6h[i-1] >= kijun_6h[i-1])
        
        if position == 0:
            # Look for entry signals: TK cross with cloud filter and volume confirmation
            # Long: TK cross up + price above cloud + volume confirmation
            long_signal = tk_cross_up and (close_val > cloud_top) and volume_confirm
            # Short: TK cross down + price below cloud + volume confirmation
            short_signal = tk_cross_down and (close_val < cloud_bottom) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. TK cross down (exit long)
            if tk_cross_down:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. TK cross up (exit short)
            if tk_cross_up:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0