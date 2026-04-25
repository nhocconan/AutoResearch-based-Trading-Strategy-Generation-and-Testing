#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike
Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above R3 with 12h EMA50 uptrend and volume > 2.5x 20-period average.
Short when price breaks below S3 with 12h EMA50 downtrend and volume > 2.5x 20-period average.
Exit on opposite band touch (R1/S1) or trend reversal.
Uses discrete sizing (0.30) to balance return and drawdown. Target: 12-37 trades/year.
Works in bull via trend-following breakouts, in bear via mean reversion at extreme bands.
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
    
    # Get 6h data for Camarilla calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 5:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Camarilla levels for each 6h bar (based on previous bar)
    R1_6h = np.full(len(close_6h), np.nan)
    S1_6h = np.full(len(close_6h), np.nan)
    R3_6h = np.full(len(close_6h), np.nan)
    S3_6h = np.full(len(close_6h), np.nan)
    R4_6h = np.full(len(close_6h), np.nan)
    S4_6h = np.full(len(close_6h), np.nan)
    
    for i in range(1, len(close_6h)):
        # Camarilla levels based on previous 6h bar's range
        high_prev = high_6h[i-1]
        low_prev = low_6h[i-1]
        close_prev = close_6h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R1_6h[i] = close_prev + (range_prev * 1.1 / 12)
            S1_6h[i] = close_prev - (range_prev * 1.1 / 12)
            R3_6h[i] = close_prev + (range_prev * 1.1 / 4)
            S3_6h[i] = close_prev - (range_prev * 1.1 / 4)
            R4_6h[i] = close_prev + (range_prev * 1.1 / 2)
            S4_6h[i] = close_prev - (range_prev * 1.1 / 2)
    
    # Align Camarilla levels to original timeframe
    R1_6h_aligned = align_htf_to_ltf(prices, df_6h, R1_6h)
    S1_6h_aligned = align_htf_to_ltf(prices, df_6h, S1_6h)
    R3_6h_aligned = align_htf_to_ltf(prices, df_6h, R3_6h)
    S3_6h_aligned = align_htf_to_ltf(prices, df_6h, S3_6h)
    R4_6h_aligned = align_htf_to_ltf(prices, df_6h, R4_6h)
    S4_6h_aligned = align_htf_to_ltf(prices, df_6h, S4_6h)
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_6h_aligned[i]) or np.isnan(S3_6h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:
            # Long: price breaks above R3 with uptrend and volume spike
            long_signal = (close[i] > R3_6h_aligned[i]) and (close[i] > ema_50_12h_aligned[i]) and vol_spike[i]
            # Short: price breaks below S3 with downtrend and volume spike
            short_signal = (close[i] < S3_6h_aligned[i]) and (close[i] < ema_50_12h_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit conditions: price touches S1 or trend reverses
            exit_signal = (close[i] < S1_6h_aligned[i]) or (close[i] < ema_50_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit conditions: price touches R1 or trend reverses
            exit_signal = (close[i] > R1_6h_aligned[i]) or (close[i] > ema_50_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0