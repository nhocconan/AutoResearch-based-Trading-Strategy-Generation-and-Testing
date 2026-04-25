#!/usr/bin/env python3
"""
6h Ichimoku Cloud + 1d Trend Filter + Volume Spike
Hypothesis: Ichimoku TK Cross (Tenkan/Kijun cross) on 6h with 1d cloud filter (price above/below 1d cloud) and volume confirmation captures strong momentum. Works in bull (long when price > 1d cloud, TK cross up) and bear (short when price < 1d cloud, TK cross down). Target 12-37 trades/year on 6h to avoid fee drag.
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
    
    # Get 1d data for cloud calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 52 for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (high_1d.rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  low_1d.rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (high_1d.rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 low_1d.rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period_senkou_b = 52
    senkou_span_b = (high_1d.rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                     low_1d.rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    
    # Chikou Span (Lagging Span): not used for signals
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Calculate ATR(14) for stop management
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Ichimoku, ATR, volume MA
    start_idx = max(52, 14, 20)  # 52 for Senkou Span B
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Trend filter: price relative to 1d cloud
        above_cloud = curr_close > cloud_top
        below_cloud = curr_close < cloud_bottom
        
        # TK Cross signals
        tk_cross_up = tenkan_val > kijun_val
        tk_cross_down = tenkan_val < kijun_val
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for TK cross signals with cloud filter and volume confirmation
            # Long: TK cross up + price above cloud + volume confirmation
            long_signal = tk_cross_up and above_cloud and volume_confirm
            # Short: TK cross down + price below cloud + volume confirmation
            short_signal = tk_cross_down and below_cloud and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: TK cross down OR price closes below cloud OR 2.0*ATR trailing stop
            if tk_cross_down or curr_close < cloud_bottom or curr_close < (highest_since_entry - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: TK cross up OR price closes above cloud OR 2.0*ATR trailing stop
            if tk_cross_up or curr_close > cloud_top or curr_close > (lowest_since_entry + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TKCross_VolumeSpike"
timeframe = "6h"
leverage = 1.0