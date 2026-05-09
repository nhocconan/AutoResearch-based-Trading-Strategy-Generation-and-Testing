#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    """
    6h Weekly Pivot R3/S3 breakout with 1d trend filter and volume confirmation.
    - Long: Close breaks above Weekly R3 with volume > 1.5x avg and price > 1d EMA(34)
    - Short: Close breaks below Weekly S3 with volume > 1.5x avg and price < 1d EMA(34)
    - Exit: Opposite breakout or price crosses back through Weekly pivot point
    - Uses weekly pivot points from previous week (excluding current)
    - Target: 15-30 trades/year on 6h timeframe (60-120 over 4 years)
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H+L+C)/3
    # R3 = Pivot + 2*(H-L), S3 = Pivot - 2*(H-L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r3 = weekly_pivot + 2 * (weekly_high - weekly_low)
    weekly_s3 = weekly_pivot - 2 * (weekly_high - weekly_low)
    weekly_pp = weekly_pivot  # pivot point for exit
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or 
            np.isnan(weekly_pp_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Close breaks above Weekly R3 with volume confirmation and above 1d EMA trend
            if close[i] > weekly_r3_aligned[i] and vol_ok and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Weekly S3 with volume confirmation and below 1d EMA trend
            elif close[i] < weekly_s3_aligned[i] and vol_ok and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close breaks below Weekly pivot point or opposite signal
            if close[i] < weekly_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close breaks above Weekly pivot point or opposite signal
            if close[i] > weekly_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals