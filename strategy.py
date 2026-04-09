#!/usr/bin/env python3
# 6h_camarilla_1d_pivot_vol
# Hypothesis: 6h Camarilla pivot levels from 1d: fade at R3/S3 (mean reversion), breakout continuation at R4/S4 (trend).
# Uses 1d pivot levels (classic + Camarilla) for structure, 60-period volume average for confirmation.
# Works in both bull/bear: mean reversion in ranges, breakouts in trends. Volume filters low-quality signals.
# Target: 15-30 trades/year (60-120 over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_1d_pivot_vol"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d classic pivot points (using previous day's OHLC)
    # We need previous day's data, so shift by 1
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Classic pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Camarilla levels
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1 / 4)
    # R2 = close + ((high - low) * 1.1 / 6)
    # R1 = close + ((high - low) * 1.1 / 12)
    # S1 = close - ((high - low) * 1.1 / 12)
    # S2 = close - ((high - low) * 1.1 / 6)
    # S3 = close - ((high - low) * 1.1 / 4)
    # S4 = close - ((high - low) * 1.1 / 2)
    high_low = prev_high - prev_low
    r4 = prev_close + (high_low * 1.1 / 2.0)
    r3 = prev_close + (high_low * 1.1 / 4.0)
    r2 = prev_close + (high_low * 1.1 / 6.0)
    r1 = prev_close + (high_low * 1.1 / 12.0)
    s1 = prev_close - (high_low * 1.1 / 12.0)
    s2 = prev_close - (high_low * 1.1 / 6.0)
    s3 = prev_close - (high_low * 1.1 / 4.0)
    s4 = prev_close - (high_low * 1.1 / 2.0)
    
    # Align pivot levels to 6h timeframe (using previous day's close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 60-period average (approx 15 days at 6h)
    vol_ma_60 = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    vol_threshold = vol_ma_60 * 1.5  # 1.5x average volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma_60[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (mean reversion failure) or below S4 (stop)
            if close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (mean reversion failure) or above R4 (stop)
            if close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price crosses above R4 with volume (breakout continuation)
            if (close[i] > r4_aligned[i]) and (volume[i] > vol_threshold[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price crosses below S4 with volume (breakout continuation)
            elif (close[i] < s4_aligned[i]) and (volume[i] > vol_threshold[i]):
                position = -1
                signals[i] = -0.25
            # Enter long: price pulls back to S3 with volume (mean reversion)
            elif (close[i] <= s3_aligned[i] * 1.005) and (close[i] >= s3_aligned[i] * 0.995) and (volume[i] > vol_threshold[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price pulls back to R3 with volume (mean reversion)
            elif (close[i] <= r3_aligned[i] * 1.005) and (close[i] >= r3_aligned[i] * 0.995) and (volume[i] > vol_threshold[i]):
                position = -1
                signals[i] = -0.25
    
    return signals