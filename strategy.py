#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter
Hypothesis: On 6h timeframe, enter long when Tenkan-sen crosses above Kijun-sen AND price is above Kumo (cloud) AND 1d EMA50 is rising AND volume > 1.3 * 20-bar average. Enter short on opposite conditions. Exit when Tenkan-sen crosses back below Kijun-sen OR price closes below/above Kumo. Uses Ichimoku from 6d HTF for more stable cloud, reducing whipsaw in ranging markets. Targets 12-30 trades/year by requiring confluence of trend, momentum, and volume.
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
    
    # Calculate Ichimoku components on 6d HTF for stability
    df_6d = get_htf_data(prices, '6d')
    if len(df_6d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_6d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_6d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_6d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_6d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_6d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_6d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align all Ichimoku components to 6h timeframe (wait for completed 6d bar)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6d, senkou_span_b)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 for Ichimoku, 50 for 1d EMA, 20 for volume
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        # Determine Kumo (cloud) boundaries
        upper_kumo = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_kumo = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Flat - look for TK cross with trend and volume confirmation
            # Long: TK cross up + price above Kumo + 1d EMA50 rising + volume filter
            tk_cross_up = (tenkan_sen_aligned[i] > kijun_sen_aligned[i]) and (tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1])
            long_entry = tk_cross_up and (close_val > upper_kumo) and (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and volume_filter[i]
            
            # Short: TK cross down + price below Kumo + 1d EMA50 falling + volume filter
            tk_cross_down = (tenkan_sen_aligned[i] < kijun_sen_aligned[i]) and (tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1])
            short_entry = tk_cross_down and (close_val < lower_kumo) and (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and volume_filter[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when TK cross down OR price closes below Kumo
            tk_cross_down = (tenkan_sen_aligned[i] < kijun_sen_aligned[i]) and (tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1])
            if tk_cross_down or (close_val < lower_kumo):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when TK cross up OR price closes above Kumo
            tk_cross_up = (tenkan_sen_aligned[i] > kijun_sen_aligned[i]) and (tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1])
            if tk_cross_up or (close_val > upper_kumo):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0