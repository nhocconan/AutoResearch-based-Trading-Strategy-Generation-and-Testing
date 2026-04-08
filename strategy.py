#!/usr/bin/env python3
# 12h_1d_camarilla_pullback_v1
# Hypothesis: Pullbacks to Camarilla pivot levels on 12h timeframe with 1d trend filter.
# Long when price touches S3 level + closes above it + 1d EMA50 up + volume confirmation.
# Short when price touches R3 level + closes below it + 1d EMA50 down + volume confirmation.
# Exit when price reaches opposite Camarilla level (S1/R1) or trend fails.
# Designed for 15-30 trades/year on 12h to avoid fee drag. Works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pullback_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Camarilla pivot levels (based on previous day)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s2 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_r2 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    
    # Calculate daily pivot points from previous day
    for i in range(1, n):
        # Previous day's OHLC
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Pivot point
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # Camarilla levels
        range_ = prev_high - prev_low
        camarilla_s3[i] = prev_close - (range_ * 1.1000 / 6)
        camarilla_s2[i] = prev_close - (range_ * 1.1000 / 4)
        camarilla_s1[i] = prev_close - (range_ * 1.1000 / 6)
        camarilla_r1[i] = prev_close + (range_ * 1.1000 / 6)
        camarilla_r2[i] = prev_close + (range_ * 1.1000 / 4)
        camarilla_r3[i] = prev_close + (range_ * 1.1000 / 6)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (24-period) for confirmation (approx 12 days)
    vol_avg = np.full(n, np.nan)
    for i in range(24, n):
        vol_avg[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 24)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_s3[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirmed = volume[i] > 1.3 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S1 level or trend fails
            if close[i] >= camarilla_s1[i] or close[i] <= ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 level or trend fails
            if close[i] <= camarilla_r1[i] or close[i] >= ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: pullback to S3 level with close above it + volume + trend filter
            if (abs(close[i] - camarilla_s3[i]) < 0.001 * camarilla_s3[i] and  # Within 0.1% of S3
                close[i] > camarilla_s3[i] and
                vol_confirmed and 
                close[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: pullback to R3 level with close below it + volume + trend filter
            elif (abs(close[i] - camarilla_r3[i]) < 0.001 * camarilla_r3[i] and  # Within 0.1% of R3
                  close[i] < camarilla_r3[i] and 
                  vol_confirmed and 
                  close[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals