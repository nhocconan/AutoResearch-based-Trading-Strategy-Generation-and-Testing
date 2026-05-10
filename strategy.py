#!/usr/bin/env python3
# 4h_Camarilla_Pivot_Reversal_Momentum
# Hypothesis: Price often reverses at Camarilla pivot levels (S3, S4, R3, R4) calculated from prior day's range.
# In both bull and bear markets, these levels act as strong support/resistance.
# Entry: Price closes beyond S3/R3 with volume confirmation and exits at S4/R4 or opposite S3/R3.
# Uses 1d timeframe for pivot calculation and 4h for execution, targeting low trade frequency.

name = "4h_Camarilla_Pivot_Reversal_Momentum"
timeframe = "4h"
leverage = 1.0

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
    # Prior day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for prior day
    # R4 = Close + ((High - Low) * 1.5)
    # R3 = Close + ((High - Low) * 1.25)
    # S3 = Close - ((High - Low) * 1.25)
    # S4 = Close - ((High - Low) * 1.5)
    range_hl = prev_high - prev_low
    r4 = prev_close + (range_hl * 1.5)
    r3 = prev_close + (range_hl * 1.25)
    s3 = prev_close - (range_hl * 1.25)
    s4 = prev_close - (range_hl * 1.5)
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation (20-period average on 4h)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # need volume MA and pivot levels
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirm = volume[i] > 1.3 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price closes below S3 (support) with volume, expect reversal up
            if close[i] < s3_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price closes above R3 (resistance) with volume, expect reversal down
            elif close[i] > r3_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches S4 (strong support) or moves back above S3
            if close[i] <= s4_aligned[i] or close[i] >= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches R4 (strong resistance) or moves back below R3
            if close[i] >= r4_aligned[i] or close[i] <= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals