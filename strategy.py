#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using 1w Camarilla pivot breakout with volume confirmation
    # Long: price breaks above weekly R4 AND volume > 1.5x 20-period average
    # Short: price breaks below weekly S4 AND volume > 1.5x 20-period average
    # Exit: price returns to weekly H3/L3 levels OR volume dry-up
    # Using weekly Camarilla pivots for structure, 6h for execution, volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly range
    weekly_range = high_1w - low_1w
    
    # Camarilla levels
    # H3/L3 = close +/- 1.1 * range / 2
    # H4/L4 = close +/- 1.5 * range / 2
    # R3/S3 = close +/- 1.1 * range
    # R4/S4 = close +/- 1.5 * range
    H3 = close_1w + 1.1 * weekly_range / 2
    L3 = close_1w - 1.1 * weekly_range / 2
    H4 = close_1w + 1.5 * weekly_range / 2
    L4 = close_1w - 1.5 * weekly_range / 2
    R3 = close_1w + 1.1 * weekly_range
    S3 = close_1w - 1.1 * weekly_range
    R4 = close_1w + 1.5 * weekly_range
    S4 = close_1w - 1.5 * weekly_range
    
    # Align weekly levels to 6h
    H3_6h = align_htf_to_ltf(prices, df_1w, H3)
    L3_6h = align_htf_to_ltf(prices, df_1w, L3)
    H4_6h = align_htf_to_ltf(prices, df_1w, H4)
    L4_6h = align_htf_to_ltf(prices, df_1w, L4)
    R3_6h = align_htf_to_ltf(prices, df_1w, R3)
    S3_6h = align_htf_to_ltf(prices, df_1w, S3)
    R4_6h = align_htf_to_ltf(prices, df_1w, R4)
    S4_6h = align_htf_to_ltf(prices, df_1w, S4)
    
    # Get 6h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H4_6h[i]) or np.isnan(L4_6h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: weekly R4/S4 breakout with volume
        long_entry = (close[i] > R4_6h[i]) and vol_confirm
        short_entry = (close[i] < S4_6h[i]) and vol_confirm
        
        # Exit logic: return to H3/L3 or volume dry-up
        long_exit = (close[i] < H3_6h[i]) or not vol_confirm
        short_exit = (close[i] > L3_6h[i]) or not vol_confirm
        
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

name = "6h_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0