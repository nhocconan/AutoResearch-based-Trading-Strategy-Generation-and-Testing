#!/usr/bin/env python3
"""
6h Ichimoku Cloud with Daily TK Cross and Volume Confirmation
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. When Tenkan-sen crosses 
above/below Kijun-sen (TK cross) with price above/below cloud (trend confirmation) and 
volume spike, it signals strong momentum continuation. Uses 6h primary with 1d HTF for 
Ichimoku calculation. Works in both bull/bear markets by following the cloud-filtered trend.
Target: 75-150 total trades over 4 years (19-37/year).
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
    
    # Get 1d data for Ichimoku calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 for Ichimoku
        return np.zeros(n)
    
    # Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = np.full_like(high_1d, np.nan)
    min_low_9 = np.full_like(low_1d, np.nan)
    for i in range(period_tenkan-1, len(high_1d)):
        max_high_9[i] = np.max(high_1d[i-(period_tenkan-1):i+1])
        min_low_9[i] = np.min(low_1d[i-(period_tenkan-1):i+1])
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = np.full_like(high_1d, np.nan)
    min_low_26 = np.full_like(low_1d, np.nan)
    for i in range(period_kijun-1, len(high_1d)):
        max_high_26[i] = np.max(high_1d[i-(period_kijun-1):i+1])
        min_low_26[i] = np.min(low_1d[i-(period_kijun-1):i+1])
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, plotted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = np.full_like(high_1d, np.nan)
    min_low_52 = np.full_like(low_1d, np.nan)
    for i in range(period_senkou_b-1, len(high_1d)):
        max_high_52[i] = np.max(high_1d[i-(period_senkou_b-1):i+1])
        min_low_52[i] = np.min(low_1d[i-(period_senkou_b-1):i+1])
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h (need 26-bar delay for Senkou spans)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)
    
    # Calculate 20-period volume MA for 6h volume confirmation
    vol_ma_20_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_6h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku and volume MA
    start_idx = max(52, 20)  # 52 for Senkou B, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        vol_ma_6h = vol_ma_20_6h[i]
        
        # Cloud boundaries (Senkou Span A and B form the cloud)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # TK cross: Tenkan-sen crossing above/below Kijun-sen
        tk_cross_up = tenkan_val > kijun_val
        tk_cross_down = tenkan_val < kijun_val
        
        # Price above/below cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Volume confirmation: current 6h volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma_6h
        
        if position == 0:
            # Look for entry signals
            # Long: TK cross up AND price above cloud AND volume confirmation
            long_entry = tk_cross_up and price_above_cloud and volume_confirm
            # Short: TK cross down AND price below cloud AND volume confirmation
            short_entry = tk_cross_down and price_below_cloud and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: TK cross down OR price falls below cloud
            if tk_cross_down or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: TK cross up OR price rises above cloud
            if tk_cross_up or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_VolumeConfirm"
timeframe = "6h"
leverage = 1.0