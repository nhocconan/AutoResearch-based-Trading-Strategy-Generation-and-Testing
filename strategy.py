#!/usr/bin/env python3
# 6h_camarilla_pivot_volume_v1
# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation.
# Long: Price breaks above R4 with volume > 1.5x 20-period average.
# Short: Price breaks below S4 with volume > 1.5x 20-period average.
# Exit: Price returns to R3/S3 level or volume divergence.
# Uses 1d pivots for institutional reference levels, volume filters weak breakouts.
# Target: 12-37 trades/year (50-150 total over 4 years) with 0.25 position size.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H/L/C from previous day
    R4 = close_1d + (high_1d - low_1d) * 1.500
    R3 = close_1d + (high_1d - low_1d) * 1.250
    S3 = close_1d - (high_1d - low_1d) * 1.250
    S4 = close_1d - (high_1d - low_1d) * 1.500
    
    # Align to 6h timeframe (1d pivots based on previous completed day)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to R3 OR volume divergence (price up but volume down)
            if close[i] <= R3_aligned[i] or (close[i] > close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to S3 OR volume divergence (price down but volume down)
            if close[i] >= S3_aligned[i] or (close[i] < close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above R4 with volume confirmation
            if (close[i] > R4_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below S4 with volume confirmation
            elif (close[i] < S4_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals