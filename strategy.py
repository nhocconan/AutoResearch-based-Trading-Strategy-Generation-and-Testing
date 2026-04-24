#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot Reversal with 1d Volume Spike Filter
- Uses 1d Camarilla levels (H3, L3, H4, L4) calculated from prior 1d candle
- Long when price crosses above H3 with volume > 2.0 * 20-period average
- Short when price crosses below L3 with volume > 2.0 * 20-period average
- Exit when price returns to H3/L3 level or opposite Camarilla level (H4/L4)
- Volume spike filter ensures institutional participation and reduces false breakouts
- Works in ranging markets (mean reversion at H3/L3) and can capture breakouts (H4/L4)
- Designed for 6h timeframe to avoid excessive trading while capturing meaningful moves
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels from prior daily candle
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on prior day's range
    range_1d = high_1d - low_1d
    close_prev = close_1d
    
    # H4, H3, L3, L4 levels
    H4 = close_prev + range_1d * 1.1 / 2
    H3 = close_prev + range_1d * 1.1 / 4
    L3 = close_prev - range_1d * 1.1 / 4
    L4 = close_prev - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (use prior day's levels)
    H4_1d = align_htf_to_ltf(prices, df_1d, H4)
    H3_1d = align_htf_to_ltf(prices, df_1d, H3)
    L3_1d = align_htf_to_ltf(prices, df_1d, L3)
    L4_1d = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 1)  # Need volume MA and at least 1 day of data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H3_1d[i]) or np.isnan(L3_1d[i]) or 
            np.isnan(H4_1d[i]) or np.isnan(L4_1d[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above H3 with volume confirmation
            if close[i] > H3_1d[i] and close[i-1] <= H3_1d[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below L3 with volume confirmation
            elif close[i] < L3_1d[i] and close[i-1] >= L3_1d[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to H3 or reaches L4 (opposite level)
            if close[i] <= H3_1d[i] or close[i] >= L4_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to L3 or reaches H4 (opposite level)
            if close[i] >= L3_1d[i] or close[i] <= H4_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0