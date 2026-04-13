#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-day data for weekly pivot levels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1-week data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Use previous week's high, low, close
    prev_week_high = df_1w['high'].shift(1).values  # Previous week's high
    prev_week_low = df_1w['low'].shift(1).values    # Previous week's low
    prev_week_close = df_1w['close'].shift(1).values # Previous week's close
    
    # Pivot point calculation
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pivot - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily volume filter: current 6h volume > 1.5x 20-period average daily volume
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Scale daily volume MA to 6h approximation (4 periods per day)
    volume_6h_approx_ma = volume_ma_20_1d_aligned / 4
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_6h_approx_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.5x 20-period average daily volume
        volume_condition = volume[i] > (volume_6h_approx_ma[i] * 1.5)
        
        # Fade at extreme pivot levels (S3/R3) with volume confirmation
        # Long when price touches or goes below S3 with volume (oversold bounce)
        # Short when price touches or goes above R3 with volume (overbought rejection)
        fade_long = close[i] <= s3_aligned[i] and volume_condition
        fade_short = close[i] >= r3_aligned[i] and volume_condition
        
        if position == 0:
            if fade_long:
                position = 1
                signals[i] = position_size
            elif fade_short:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price returns to pivot level or shows reversal
            if close[i] >= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price returns to pivot level or shows reversal
            if close[i] <= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_Fade_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0