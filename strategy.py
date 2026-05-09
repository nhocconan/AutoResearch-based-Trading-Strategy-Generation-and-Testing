#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivotBreakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (weekly high, low, close)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Weekly high, low, close for pivot calculation
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    
    # Calculate weekly pivot and support/resistance levels
    # Pivot = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1 / 12
    # S1 = Pivot - (H - L) * 1.1 / 12
    # R2 = Pivot + (H - L) * 1.1 / 6
    # S2 = Pivot - (H - L) * 1.1 / 6
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    weekly_r1 = weekly_pivot + weekly_range * 1.1 / 12
    weekly_s1 = weekly_pivot - weekly_range * 1.1 / 12
    weekly_r2 = weekly_pivot + weekly_range * 1.1 / 6
    weekly_s2 = weekly_pivot - weekly_range * 1.1 / 6
    
    # Align weekly levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_w, weekly_s2)
    
    # Daily EMA(34) for trend filter
    close_d = pd.Series(df_d['close'])
    ema34_d = close_d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(weekly_r2_aligned[i]) or np.isnan(weekly_s2_aligned[i]) or
            np.isnan(ema34_d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        
        if position == 0:
            # Long: Price breaks above weekly R2 with volume and above daily EMA trend
            if close[i] > weekly_r2_aligned[i] and vol_ok and close[i] > ema34_d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S2 with volume and below daily EMA trend
            elif close[i] < weekly_s2_aligned[i] and vol_ok and close[i] < ema34_d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly S1 (reversion to mean)
            if close[i] < weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly R1 (reversion to mean)
            if close[i] > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals