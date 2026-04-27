#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm
Hypothesis: Uses weekly Camarilla R3/S3 levels for breakout entries aligned with 1-week trend (EMA50).
Enter long when price breaks above weekly R3 AND weekly close > weekly EMA50 AND volume spike.
Enter short when price breaks below weekly S3 AND weekly close < weekly EMA50 AND volume spike.
Exit when price returns to the Camarilla level or weekly trend reverses.
Designed for 1d timeframe to achieve 30-100 total trades over 4 years.
Weekly Camarilla levels provide institutional support/resistance; volume confirms participation.
Works in both bull and bear markets by following weekly trend while using Camarilla levels for precise breakout entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    # R3 = H + 2*(H-L)/1.1, S3 = L - 2*(H-L)/1.1
    h_prev = df_1w['high'].shift(1).values  # Previous week high
    l_prev = df_1w['low'].shift(1).values   # Previous week low
    c_prev = df_1w['close'].shift(1).values # Previous week close
    
    # Avoid division by zero and ensure valid calculations
    hl_range = h_prev - l_prev
    r3 = h_prev + (2 * hl_range / 1.1)
    s3 = l_prev - (2 * hl_range / 1.1)
    
    # Weekly EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align all weekly data to daily timeframe (no look-ahead)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need weekly EMA50 (50), volume avg (20), and previous week data
    start_idx = max(50, 20) + 1  # +1 for weekly shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of weekly Camarilla level with weekly trend filter AND volume
            # Long: price breaks above weekly R3 AND weekly uptrend AND volume spike
            long_condition = (close_val > r3_val) and (close_val > ema_val) and vol_conf
            # Short: price breaks below weekly S3 AND weekly downtrend AND volume spike
            short_condition = (close_val < s3_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to weekly R3 level OR weekly trend breaks
            exit_condition = (close_val <= r3_val) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to weekly S3 level OR weekly trend breaks
            exit_condition = (close_val >= s3_val) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0