#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Reversal_1dTrendFilter_VolumeSpike_v1
Hypothesis: 6h mean reversion at Camarilla R3/S3 levels with 1d trend filter and volume spike confirmation.
- Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year) to minimize fee drag
- Camarilla levels calculated from prior 1d session (HLC) for R3/S3 mean reversion zones
- 1d EMA50 filter ensures trades align with higher timeframe trend (bull/bear agnostic)
- Volume spike (>1.5x 20-period average) confirms institutional interest at reversal points
- Long at S3 in uptrend with volume spike, Short at R3 in downtrend with volume spike
- Designed to work in both bull and bear markets by trading reversals within the prevailing 1d trend
- Exit on Tenkan/Kijun crossover or opposite Camarilla level touch (R1/S1)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from prior 1d session (HLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We'll calculate for each 1d bar and align to 6h
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_r1_1d = close_1d + 1.0 * (high_1d - low_1d)  # For exit
    camarilla_s1_1d = close_1d - 1.0 * (high_1d - low_1d)  # For exit
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # Volume spike confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla conditions
        at_r3 = abs(close[i] - r3_aligned[i]) < (0.002 * close[i])  # Within 0.2% of R3
        at_s3 = abs(close[i] - s3_aligned[i]) < (0.002 * close[i])  # Within 0.2% of S3
        at_r1 = abs(close[i] - r1_aligned[i]) < (0.002 * close[i])  # Within 0.2% of R1 (exit long)
        at_s1 = abs(close[i] - s1_aligned[i]) < (0.002 * close[i])  # Within 0.2% of S1 (exit short)
        
        # Trend filter
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: at S3 in uptrend with volume spike
            if at_s3 and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: at R3 in downtrend with volume spike
            elif at_r3 and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price reaches R1 (take profit) or volume dies
            if at_r1 or not vol_spike:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price reaches S1 (take profit) or volume dies
            if at_s1 or not vol_spike:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Reversal_1dTrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0