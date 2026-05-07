#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_WithVolume
Hypothesis: Camarilla R3/S3 levels from 1-day act as strong support/resistance. 
Breakout above R3 or below S3 with 1-day EMA trend filter and volume confirmation 
captures genuine momentum moves. Fades at R4/S4 to avoid overextension. 
Works in bull/bear via trend filter. Target: 50-150 total trades over 4 years.
"""
name = "6h_Camarilla_R3S3_Breakout_1dTrend_WithVolume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1-day data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We'll use vectorized calculation for efficiency
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1-day bar
    H_L = high_1d - low_1d
    R3 = close_1d + (H_L * 1.1 / 4)
    S3 = close_1d - (H_L * 1.1 / 4)
    R4 = close_1d + (H_L * 1.1 / 2)
    S4 = close_1d - (H_L * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (wait for daily close)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with uptrend + volume
            if close[i] > R3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with downtrend + volume
            elif close[i] < S3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long: exit at R4 (take profit) or if trend fails
            if close[i] >= R4_aligned[i]:
                signals[i] = 0.0  # Take profit at R4
                position = 0
            elif close[i] < ema_34_1d_aligned[i]:  # Trend failure
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short: exit at S4 (take profit) or if trend fails
            if close[i] <= S4_aligned[i]:
                signals[i] = 0.0  # Take profit at S4
                position = 0
            elif close[i] > ema_34_1d_aligned[i]:  # Trend failure
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals