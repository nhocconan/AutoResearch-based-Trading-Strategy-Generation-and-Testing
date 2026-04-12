#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot long/short with 1d volume spike filter
    # Camarilla levels provide mean-reversion edges in ranging markets
    # Volume spike confirms institutional interest at pivot touches
    # Works in bull/bear by fading extremes with volume validation
    # Target: 12-37 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume context (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # H4 = Pivot + 1.1 * (H - L) / 2
    # L4 = Pivot - 1.1 * (H - L) / 2
    # H3 = Pivot + 1.1 * (H - L) / 4
    # L3 = Pivot - 1.1 * (H - L) / 4
    # H2 = Pivot + 1.1 * (H - L) / 6
    # L2 = Pivot - 1.1 * (H - L) / 6
    # H1 = Pivot + 1.1 * (H - L) / 12
    # L1 = Pivot - 1.1 * (H - L) / 12
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    h4_1d = pivot_1d + 1.1 * range_1d / 2.0
    l4_1d = pivot_1d - 1.1 * range_1d / 2.0
    h3_1d = pivot_1d + 1.1 * range_1d / 4.0
    l3_1d = pivot_1d - 1.1 * range_1d / 4.0
    h2_1d = pivot_1d + 1.1 * range_1d / 6.0
    l2_1d = pivot_1d - 1.1 * range_1d / 6.0
    h1_1d = pivot_1d + 1.1 * range_1d / 12.0
    l1_1d = pivot_1d - 1.1 * range_1d / 12.0
    
    # Align 1d Camarilla levels to 12h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h2_1d_aligned = align_htf_to_ltf(prices, df_1d, h2_1d)
    l2_1d_aligned = align_htf_to_ltf(prices, df_1d, l2_1d)
    h1_1d_aligned = align_htf_to_ltf(prices, df_1d, h1_1d)
    l1_1d_aligned = align_htf_to_ltf(prices, df_1d, l1_1d)
    
    # 1d volume spike filter: current volume > 1.5 * 20-period average
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = (not np.isnan(vol_ma_20_1d_aligned[i]) and 
                   volume_1d[-1] > 1.5 * vol_ma_20_1d_aligned[i]) if len(volume_1d) > 0 else False
    
    # For per-bar volume spike, we need to calculate it properly
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or 
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_filter = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Mean reversion at Camarilla extremes with volume confirmation
        # Long near L3/L4, Short near H3/H4
        long_entry = ((close[i] <= l3_1d_aligned[i] or close[i] <= l4_1d_aligned[i]) and 
                     vol_filter)
        short_entry = ((close[i] >= h3_1d_aligned[i] or close[i] >= h4_1d_aligned[i]) and 
                      vol_filter)
        
        # Exit when price moves back toward pivot (mean reversion complete)
        long_exit = close[i] >= pivot_1d_aligned[i] if not np.isnan(pivot_1d_aligned[i]) else False
        short_exit = close[i] <= pivot_1d_aligned[i] if not np.isnan(pivot_1d_aligned[i]) else False
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_mean_reversion_vol_filter_v1"
timeframe = "12h"
leverage = 1.0