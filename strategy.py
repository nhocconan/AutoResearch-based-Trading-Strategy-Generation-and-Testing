#!/usr/bin/env python3
name = "1d_Weekly_Camarilla_Pivot_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels: H, L, C from previous week
    H = df_1w['high'].values
    L = df_1w['low'].values
    C = df_1w['close'].values
    
    # Camarilla levels: H3, L3, H4, L4
    # H3 = C + (H - L) * 1.1 / 4
    # L3 = C - (H - L) * 1.1 / 4
    # H4 = C + (H - L) * 1.1 / 2
    # L4 = C - (H - L) * 1.1 / 2
    range_w = H - L
    H3 = C + range_w * 1.1 / 4
    L3 = C - range_w * 1.1 / 4
    H4 = C + range_w * 1.1 / 2
    L4 = C - range_w * 1.1 / 2
    
    # Align weekly Camarilla levels to daily timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1w, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1w, L4)
    
    # Volume confirmation: current volume > 1.5 * 20-day average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches or crosses above L3 with volume confirmation
            if close[i] >= L3_aligned[i] and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches or crosses below H3 with volume confirmation
            elif close[i] <= H3_aligned[i] and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches or crosses H4 (resistance) or closes below L3
            if close[i] >= H4_aligned[i] or close[i] < L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches or crosses L4 (support) or closes above H3
            if close[i] <= L4_aligned[i] or close[i] > H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals