#!/usr/bin/env python3
"""
6H_WeeklyPivot_R3S3_R4S4_Strategy
Trades breakouts of weekly R3/S3 for continuation and R4/S4 for reversals.
Long when price breaks above weekly R3 with volume confirmation and weekly trend up.
Short when price breaks below weekly S3 with volume confirmation and weekly trend down.
Exit when price returns to weekly pivot (R1/S1 level) or opposite S3/R3 level is breached.
Uses weekly pivots for institutional levels, volume for confirmation, and weekly trend filter.
Targets 15-35 trades per year on 6H timeframe.
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
    
    # Get weekly data for pivot points and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # R1, S1
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # R2, S2
    weekly_r2 = weekly_pivot + weekly_range
    weekly_s2 = weekly_pivot - weekly_range
    
    # R3, S3 (primary breakout levels)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # R4, S4 (extreme reversal levels)
    weekly_r4 = weekly_r3 + weekly_range
    weekly_s4 = weekly_s3 - weekly_range
    
    # Weekly trend filter: price above/below weekly pivot
    weekly_trend_up = weekly_close > weekly_pivot
    weekly_trend_down = weekly_close < weekly_pivot
    
    # Align weekly levels to 6H timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s4)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_down.astype(float))
    
    # Volume confirmation: 20-period volume average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need volume MA20
    start_idx = 19
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(weekly_s3_aligned[i]) or np.isnan(weekly_r4_aligned[i]) or 
            np.isnan(weekly_s4_aligned[i]) or np.isnan(weekly_trend_up_aligned[i]) or 
            np.isnan(weekly_trend_down_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: break above weekly R3 with volume and weekly uptrend
            if (price > weekly_r3_aligned[i] and 
                weekly_trend_up_aligned[i] > 0.5 and 
                vol_filter):
                signals[i] = size
                position = 1
            # Short: break below weekly S3 with volume and weekly downtrend
            elif (price < weekly_s3_aligned[i] and 
                  weekly_trend_down_aligned[i] > 0.5 and 
                  vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: return to weekly pivot or break below S3 (reversal)
            if (price < weekly_pivot_aligned[i] or 
                price < weekly_s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: return to weekly pivot or break above R3 (reversal)
            if (price > weekly_pivot_aligned[i] or 
                price > weekly_r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6H_WeeklyPivot_R3S3_R4S4_Strategy"
timeframe = "6h"
leverage = 1.0