#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d timeframe
# Uses Camarilla pivot calculation (R1-S1, R2-S2, R3-S3, R4-S4) from daily candles
# Fades at R3/S3 (mean reversion) and breaks out at R4/S4 (trend continuation)
# Requires volume confirmation (>1.5x 20-bar average) to avoid false signals
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear: captures reversals at extreme levels and breakouts in strong moves

name = "6h_Camarilla_R3S3_R4S4_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    # R4 = close + (high - low) * 1.5
    # R3 = close + (high - low) * 1.25
    # R2 = close + (high - low) * 1.166
    # R1 = close + (high - low) * 1.083
    # S1 = close - (high - low) * 1.083
    # S2 = close - (high - low) * 1.166
    # S3 = close - (high - low) * 1.25
    # S4 = close - (high - low) * 1.5
    high_low_range = high_prev - low_prev
    r4 = close_prev + high_low_range * 1.5
    r3 = close_prev + high_low_range * 1.25
    r2 = close_prev + high_low_range * 1.166
    r1 = close_prev + high_low_range * 1.083
    s1 = close_prev - high_low_range * 1.083
    s2 = close_prev - high_low_range * 1.166
    s3 = close_prev - high_low_range * 1.25
    s4 = close_prev - high_low_range * 1.5
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long fade at S3: price touches or goes below S3 but closes back above it
            if close[i] <= s3_aligned[i] and close[i] > s2_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short fade at R3: price touches or goes above R3 but closes back below it
            elif close[i] >= r3_aligned[i] and close[i] < r2_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            # Long breakout at R4: price breaks above R4 with volume
            elif close[i] > r4_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown at S4: price breaks below S4 with volume
            elif close[i] < s4_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches R1 (profit target) or S2 (stop loss)
            if close[i] >= r1_aligned[i] or close[i] <= s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches S1 (profit target) or R2 (stop loss)
            if close[i] <= s1_aligned[i] or close[i] >= r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals