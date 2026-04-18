#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot (from prior week's data)
    # Resample daily to weekly manually by grouping 5-day periods
    weeks_high = []
    weeks_low = []
    weeks_close = []
    
    for i in range(0, len(high_1d), 5):
        end = min(i + 5, len(high_1d))
        weeks_high.append(np.max(high_1d[i:end]))
        weeks_low.append(np.min(low_1d[i:end]))
        weeks_close.append(close_1d[end-1])
    
    # Calculate weekly pivot points
    weeks_high = np.array(weeks_high)
    weeks_low = np.array(weeks_low)
    weeks_close = np.array(weeks_close)
    
    # Pivot = (H + L + C)/3
    weekly_pivot = (weeks_high + weeks_low + weeks_close) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - weeks_low
    weekly_s1 = 2 * weekly_pivot - weeks_high
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_r2 = weekly_pivot + (weeks_high - weeks_low)
    weekly_s2 = weekly_pivot - (weeks_high - weeks_low)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_r3 = weeks_high + 2 * (weekly_pivot - weeks_low)
    weekly_s3 = weeks_low - 2 * (weeks_high - weeks_pivot)
    
    # Align weekly levels to 6h timeframe (with 1-week delay for confirmation)
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot, additional_delay_bars=5)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1, additional_delay_bars=5)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1, additional_delay_bars=5)
    weekly_r2_6h = align_htf_to_ltf(prices, df_1d, weekly_r2, additional_delay_bars=5)
    weekly_s2_6h = align_htf_to_ltf(prices, df_1d, weekly_s2, additional_delay_bars=5)
    weekly_r3_6h = align_htf_to_ltf(prices, df_1d, weekly_r3, additional_delay_bars=5)
    weekly_s3_6h = align_htf_to_ltf(prices, df_1d, weekly_s3, additional_delay_bars=5)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirmed = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_6h[i]) or np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or np.isnan(weekly_r2_6h[i]) or 
            np.isnan(weekly_s2_6h[i]) or np.isnan(weekly_r3_6h[i]) or 
            np.isnan(weekly_s3_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly R3 with volume
            if close[i] > weekly_r3_6h[i] and vol_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S3 with volume
            elif close[i] < weekly_s3_6h[i] and vol_confirmed[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price falls back below weekly pivot
            if close[i] < weekly_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above weekly pivot
            if close[i] > weekly_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_Volume"
timeframe = "6h"
leverage = 1.0