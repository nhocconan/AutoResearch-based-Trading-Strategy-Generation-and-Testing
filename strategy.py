#!/usr/bin/env python3
"""
Hypothesis: 6-hour strategy using weekly pivot levels with daily EMA trend filter.
Long when price breaks above weekly R3 with daily EMA(50) uptrend and volume surge.
Short when price breaks below weekly S3 with daily EMA(50) downtrend and volume surge.
Exit when price returns to weekly pivot point (P) or EMA trend reverses.
Designed for low turnover: ~15-25 trades/year per symbol to minimize fee drag.
Uses weekly structure for direction and daily EMA for trend filter to avoid whipsaws.
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
    
    # Load weekly data once for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points (based on previous week)
    # P = (H + L + C) / 3
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    weekly_P = (high_weekly + low_weekly + close_weekly) / 3
    weekly_R3 = high_weekly + 2 * (weekly_P - low_weekly)
    weekly_S3 = low_weekly - 2 * (high_weekly - weekly_P)
    
    # Load daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    # EMA(50) for trend filter
    ema_50 = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 24-period average (4 days for 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Weekly index (approximately 28 bars per week for 6h timeframe)
        idx_weekly = i // 28
        if idx_weekly < 1:  # need previous week for pivot
            continue
        
        # Daily index (4 bars per day for 6h timeframe)
        idx_daily = i // 4
        if idx_daily < 50:  # need enough for EMA
            continue
        
        # Use previous weekly values to avoid look-ahead
        prev_weekly_idx = idx_weekly - 1
        if prev_weekly_idx < 0:
            continue
            
        # Use current daily values for EMA (updated daily)
        # EMA uses all history up to current point, so current value is valid
        
        P_weekly = weekly_P[prev_weekly_idx] if prev_weekly_idx < len(weekly_P) else weekly_P[-1]
        R3_weekly = weekly_R3[prev_weekly_idx] if prev_weekly_idx < len(weekly_R3) else weekly_R3[-1]
        S3_weekly = weekly_S3[prev_weekly_idx] if prev_weekly_idx < len(weekly_S3) else weekly_S3[-1]
        
        if np.isnan(P_weekly) or np.isnan(R3_weekly) or np.isnan(S3_weekly):
            continue
        
        # Create arrays for alignment (using previous weekly values)
        P_arr = np.full(len(df_weekly), P_weekly)
        R3_arr = np.full(len(df_weekly), R3_weekly)
        S3_arr = np.full(len(df_weekly), S3_weekly)
        
        P_6h = align_htf_to_ltf(prices, df_weekly, P_arr)[i]
        R3_6h = align_htf_to_ltf(prices, df_weekly, R3_arr)[i]
        S3_6h = align_htf_to_ltf(prices, df_weekly, S3_arr)[i]
        
        # Daily EMA value (no alignment needed as it's updated daily)
        ema_50_6h = ema_50[idx_daily] if idx_daily < len(ema_50) else ema_50[-1]
        
        if np.isnan(P_6h) or np.isnan(R3_6h) or np.isnan(S3_6h) or np.isnan(ema_50_6h):
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3 + daily EMA uptrend + volume surge
            if (close[i] > R3_6h and 
                close[i] > ema_50_6h and  # price above EMA = uptrend
                volume[i] > vol_ma[i] * 2.0):
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly S3 + daily EMA downtrend + volume surge
            elif (close[i] < S3_6h and 
                  close[i] < ema_50_6h and  # price below EMA = downtrend
                  volume[i] > vol_ma[i] * 2.0):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price returns to weekly pivot or price breaks below EMA
            if close[i] < P_6h or close[i] < ema_50_6h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price returns to weekly pivot or price breaks above EMA
            if close[i] > P_6h or close[i] > ema_50_6h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_1w_Pivot_1d_EMA50_Volume"
timeframe = "6h"
leverage = 1.0
EOF