#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high/low/close for pivot points
    n_days = len(high_1d)
    weekly_high = np.full(n_days, np.nan)
    weekly_low = np.full(n_days, np.nan)
    weekly_close = np.full(n_days, np.nan)
    
    # Weekly aggregation: group by week (Monday-Sunday)
    for i in range(n_days):
        # Find start of week (most recent Monday)
        # Simplified: use 5-day lookback for weekly calculation
        start_idx = max(0, i - 4)  # Last 5 days including current
        weekly_high[i] = np.max(high_1d[start_idx:i+1])
        weekly_low[i] = np.min(low_1d[start_idx:i+1])
        weekly_close[i] = close_1d[i]
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + weekly_range
    weekly_s2 = weekly_pivot - weekly_range
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivots to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # Calculate 6h ATR for volatility filter
    tr_6h_1 = high - low
    tr_6h_2 = np.abs(high - np.roll(close, 1))
    tr_6h_3 = np.abs(low - np.roll(close, 1))
    tr_6h_1[0] = high[0] - low[0]
    tr_6h_2[0] = np.abs(high[0] - close[0])
    tr_6h_3[0] = np.abs(low[0] - close[0])
    tr_6h = np.maximum(tr_6h_1, np.maximum(tr_6h_2, tr_6h_3))
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need volume MA and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(weekly_s3_aligned[i]) or np.isnan(atr_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above weekly R2 with volume
            if (close[i] > weekly_r2_aligned[i] and vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S2 with volume
            elif (close[i] < weekly_s2_aligned[i] and vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below weekly pivot or ATR stop
            if (close[i] < weekly_pivot_aligned[i] or 
                close[i] < weekly_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly pivot or ATR stop
            if (close[i] > weekly_pivot_aligned[i] or 
                close[i] > weekly_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R2S2_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0