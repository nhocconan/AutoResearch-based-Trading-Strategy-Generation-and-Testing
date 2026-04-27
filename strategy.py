#!/usr/bin/env python3
"""
6h_WeeklyPivot_R3_S3_Breakout_Trend_Filter
Hypothesis: Weekly pivots capture longer-term market structure. Breakouts at weekly R3/S3 with trend filter (EMA50) and volume confirmation work in both bull and bear markets by capturing continuation moves after consolidation. Weekly timeframe reduces noise, and volume confirmation filters false breakouts.
Timeframe: 6h
Weekly pivot levels calculated from prior week's OHLC.
Entry: Long when price > weekly R3 + volume spike + price > EMA50; Short when price < weekly S3 + volume spike + price < EMA50
Exit: Price crosses back through pivot level or trend fails
Position size: 0.25 (discrete to minimize churn)
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly R3 and S3 levels (using prior week's OHLC)
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    weekly_r3 = np.zeros(len(close_weekly))
    weekly_s3 = np.zeros(len(close_weekly))
    
    for i in range(len(close_weekly)):
        if i == 0:
            weekly_r3[i] = close_weekly[i]
            weekly_s3[i] = close_weekly[i]
        else:
            # Use previous week's OHLC for current week's levels
            ph = high_weekly[i-1]
            pl = low_weekly[i-1]
            pc = close_weekly[i-1]
            weekly_r3[i] = pc + (ph - pl) * 1.1 / 2
            weekly_s3[i] = pc - (ph - pl) * 1.1 / 2
    
    # Align weekly levels to 6h timeframe
    weekly_r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    
    # Calculate EMA50 on weekly for trend filter
    ema_weekly_period = 50
    ema_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= ema_weekly_period:
        ema_weekly[ema_weekly_period - 1] = np.mean(close_weekly[:ema_weekly_period])
        for i in range(ema_weekly_period, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * (2 / (ema_weekly_period + 1)) + 
                             ema_weekly[i-1] * (1 - (2 / (ema_weekly_period + 1))))
    
    # Align EMA50 to 6h timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate volume spike (current volume vs 20-period average)
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    if n >= vol_ma_period:
        for i in range(vol_ma_period - 1, n):
            vol_ma[i] = np.mean(volume[i - vol_ma_period + 1:i + 1])
    
    volume_spike = np.full(n, False)
    for i in range(vol_ma_period - 1, n):
        if vol_ma[i] > 0:
            volume_spike[i] = volume[i] > (vol_ma[i] * 1.5)  # 50% above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivots, EMA50, and volume MA
    start_idx = max(50, vol_ma_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or 
            np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r3_level = weekly_r3_aligned[i]
        s3_level = weekly_s3_aligned[i]
        ema_trend = ema_weekly_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and uptrend (price > EMA50)
            if (price > r3_level and vol_spike and price > ema_trend):
                signals[i] = size
                position = 1
            # Short: Price breaks below S3 with volume spike and downtrend (price < EMA50)
            elif (price < s3_level and vol_spike and price < ema_trend):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price returns below R3 or trend fails
            if price < r3_level or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price returns above S3 or trend fails
            if price > s3_level or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_R3_S3_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0