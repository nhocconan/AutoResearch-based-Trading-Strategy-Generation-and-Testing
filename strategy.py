#!/usr/bin/env python3
name = "1d_Weekly_Camarilla_Pivot_Volume"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on previous week's high, low, close)
    # Camarilla levels: 
    # H4 = close + 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4
    # H2 = close + 1.1*(high-low)/6
    # H1 = close + 1.1*(high-low)/12
    # L1 = close - 1.1*(high-low)/12
    # L2 = close - 1.1*(high-low)/6
    # L3 = close - 1.1*(high-low)/4
    # L4 = close - 1.1*(high-low)/2
    
    # Shift by 1 to use previous week's data (no look-ahead)
    high_1w = df_1w['high'].shift(1).values
    low_1w = df_1w['low'].shift(1).values
    close_1w = df_1w['close'].shift(1).values
    
    # Calculate Camarilla levels
    H4 = close_1w + 1.1 * (high_1w - low_1w) / 2
    H3 = close_1w + 1.1 * (high_1w - low_1w) / 4
    H2 = close_1w + 1.1 * (high_1w - low_1w) / 6
    H1 = close_1w + 1.1 * (high_1w - low_1w) / 12
    L1 = close_1w - 1.1 * (high_1w - low_1w) / 12
    L2 = close_1w - 1.1 * (high_1w - low_1w) / 6
    L3 = close_1w - 1.1 * (high_1w - low_1w) / 4
    L4 = close_1w - 1.1 * (high_1w - low_1w) / 2
    
    # Align weekly levels to daily timeframe (wait for weekly bar to close)
    H4_aligned = align_htf_to_ltf(prices, df_1w, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3)
    H2_aligned = align_htf_to_ltf(prices, df_1w, H2)
    H1_aligned = align_htf_to_ltf(prices, df_1w, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1w, L1)
    L2_aligned = align_htf_to_ltf(prices, df_1w, L2)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1w, L4)
    
    # Calculate weekly volume average for volume confirmation
    vol_1w = df_1w['volume'].shift(1).values  # Previous week's volume
    vol_ma4 = pd.Series(vol_1w).rolling(window=4, min_periods=4).mean().values  # 4-week average
    vol_ma4_aligned = align_htf_to_ltf(prices, df_1w, vol_ma4)
    
    # Daily volume for comparison
    vol_ma4_daily = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for weekly data
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(vol_ma4_aligned[i]) or np.isnan(vol_ma4_daily[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current 4-day average volume > 4-week average volume
        volume_condition = vol_ma4_daily[i] > vol_ma4_aligned[i]
        
        if position == 0:
            # Long: price touches or goes below L3 level with volume confirmation
            if (low[i] <= L3_aligned[i] and volume_condition):
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above H3 level with volume confirmation
            elif (high[i] >= H3_aligned[i] and volume_condition):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above H3 level (reversal signal)
            if high[i] >= H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below L3 level (reversal signal)
            if low[i] <= L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals