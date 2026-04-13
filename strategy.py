#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout
Hypothesis: Uses 1d Camarilla pivot levels as support/resistance with 4h breakouts and volume confirmation.
Camarilla levels work across bull/bear markets by identifying key reversal/breakout points.
Combined with volume confirmation and a loose trend filter (price > 50-period SMA) to avoid false signals.
Target: 20-40 trades/year on 4h (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # Formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    H = high_1d
    L = low_1d
    C = close_1d
    range_hl = H - L
    
    # Camarilla levels (using previous day's data)
    R4 = C + (range_hl * 1.1 / 2)
    R3 = C + (range_hl * 1.1 / 4)
    R2 = C + (range_hl * 1.1 / 6)
    R1 = C + (range_hl * 1.1 / 12)
    S1 = C - (range_hl * 1.1 / 12)
    S2 = C - (range_hl * 1.1 / 6)
    S3 = C - (range_hl * 1.1 / 4)
    S4 = C - (range_hl * 1.1 / 2)
    
    # Shift levels to avoid look-ahead (use previous day's levels)
    R4 = np.roll(R4, 1)
    R3 = np.roll(R3, 1)
    R2 = np.roll(R2, 1)
    R1 = np.roll(R1, 1)
    S1 = np.roll(S1, 1)
    S2 = np.roll(S2, 1)
    S3 = np.roll(S3, 1)
    S4 = np.roll(S4, 1)
    R4[0] = R3[0] = R2[0] = R1[0] = S1[0] = S2[0] = S3[0] = S4[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 4h trend filter: price > 50-period SMA (avoid shorting strong uptrends)
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean()
    trend_filter = close > sma_50
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_confirm = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(R2_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or np.isnan(trend_filter[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above R3 with volume and trend filter
        if close[i] > R3_aligned[i] and volume_confirm[i] and trend_filter[i]:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        
        # Short entry: price breaks below S3 (no trend filter for shorts to work in bear markets)
        elif close[i] < S3_aligned[i] and volume_confirm[i]:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        
        # Exit conditions: price returns to median levels (S1/R1) or opposite Camarilla level
        elif position == 1 and (close[i] <= S1_aligned[i] or close[i] >= R1_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= S1_aligned[i] or close[i] <= R1_aligned[i]):
            position = 0
            signals[i] = 0.0
        
        # Hold position
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Camarilla_Pivot_Breakout"
timeframe = "4h"
leverage = 1.0