#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Price tends to reverse from Camarilla pivot levels (R1/S1) derived from
1-day data when aligned with the 1-day trend (EMA34) and confirmed by volume spikes.
Long when price pulls back to S1 in an uptrend with volume confirmation.
Short when price rallies to R1 in a downtrend with volume confirmation.
This structure provides high-probability reversals in both bull and bear markets
while keeping trade frequency low (target: 50-150 trades over 4 years).
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Camarilla pivot levels from 1-day data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_pivot = typical_price_1d
    r1_1d = camarilla_pivot + (range_1d * 1.1 / 12)
    s1_1d = camarilla_pivot - (range_1d * 1.1 / 12)
    r2_1d = camarilla_pivot + (range_1d * 1.1 / 6)
    s2_1d = camarilla_pivot - (range_1d * 1.1 / 6)
    r3_1d = camarilla_pivot + (range_1d * 1.1 / 4)
    s3_1d = camarilla_pivot - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1)  # Need EMA34 and at least one Camarilla level
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or \
           np.isnan(camarilla_pivot_aligned[i]) or np.isnan(r1_1d_aligned[i]) or \
           np.isnan(s1_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or \
           np.isnan(s2_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 1d volume (scaled to 12h)
        # 2x 12h periods in 1d
        vol_12h_approx = vol_sma20_1d_aligned[i] / 2.0
        volume_confirm = volume[i] > 1.5 * vol_12h_approx
        
        if position == 0:
            # Long: Price near S1/S2 in uptrend with volume confirmation
            # Allow 0.3% buffer around pivot levels
            near_s1 = abs(close[i] - s1_1d_aligned[i]) / s1_1d_aligned[i] < 0.003
            near_s2 = abs(close[i] - s2_1d_aligned[i]) / s2_1d_aligned[i] < 0.003
            if (near_s1 or near_s2) and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price near R1/R2 in downtrend with volume confirmation
            elif (abs(close[i] - r1_1d_aligned[i]) / r1_1d_aligned[i] < 0.003 or
                  abs(close[i] - r2_1d_aligned[i]) / r2_1d_aligned[i] < 0.003) and \
                 close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price reaches pivot or R1, or trend reversal
            if (close[i] >= camarilla_pivot_aligned[i] or 
                close[i] >= r1_1d_aligned[i] or
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price reaches pivot or S1, or trend reversal
            if (close[i] <= camarilla_pivot_aligned[i] or 
                close[i] <= s1_1d_aligned[i] or
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals