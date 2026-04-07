#!/usr/bin/env python3
"""
12h Camarilla Pivot + 1d Volume Spike + Weekly Trend Filter
Long when price touches S3 support with volume spike and weekly trend up
Short when price touches R3 resistance with volume spike and weekly trend down
Exit when price crosses median line or weekly trend reverses
Uses weekly trend to filter direction, Camarilla levels from 1d for entries
Designed for low-frequency, high-conviction trades in all market regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Trend Filter (EMA crossover) ===
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === Daily Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for current day using previous day's data
    range_ = prev_high - prev_low
    # Avoid division by zero
    range_ = np.where(range_ == 0, 1, range_)
    
    # Camarilla multipliers
    S1 = prev_close - (range_ * 1.1 / 12)
    S2 = prev_close - (range_ * 1.1 / 6)
    S3 = prev_close - (range_ * 1.1 / 4)
    R1 = prev_close + (range_ * 1.1 / 12)
    R2 = prev_close + (range_ * 1.1 / 6)
    R3 = prev_close + (range_ * 1.1 / 4)
    
    # Align to 12h timeframe (Camarilla levels are constant throughout the day)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    # Midpoint (Pivot) for exit
    PP = prev_close
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    
    # === Daily Volume Spike Filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(PP_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend direction
        weekly_up = ema_50_1w_aligned[i] > ema_200_1w_aligned[i]
        weekly_down = ema_50_1w_aligned[i] < ema_200_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses above pivot (PP) or weekly trend turns down
            if close[i] > PP_aligned[i] or not weekly_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses below pivot (PP) or weekly trend turns up
            if close[i] < PP_aligned[i] or not weekly_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade in direction of weekly trend
            if weekly_up and vol_spike[i]:
                # Long when price touches S3 support
                if close[i] <= S3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
            elif weekly_down and vol_spike[i]:
                # Short when price touches R3 resistance
                if close[i] >= R3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals