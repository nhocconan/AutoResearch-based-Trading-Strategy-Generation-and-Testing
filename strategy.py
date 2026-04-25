#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirm
Hypothesis: Daily Camarilla R1/S1 breakout with 1-week EMA50 trend filter and volume spike confirmation.
Long when price breaks above R1 with 1w EMA50 uptrend and volume > 2.0x 20-period average.
Short when price breaks below S1 with 1w EMA50 downtrend and volume > 2.0x 20-period average.
Exit on opposite band touch (S1/R1) or trend reversal.
Uses discrete sizing (0.25) to minimize fee churn. Target: 7-25 trades/year.
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
    
    # Get 1d data for Camarilla calculations (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (based on previous bar)
    R1_1d = np.full(len(close_1d), np.nan)
    S1_1d = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Camarilla levels based on previous 1d bar's range
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R1_1d[i] = close_prev + (range_prev * 1.1 / 12)
            S1_1d[i] = close_prev - (range_prev * 1.1 / 12)
    
    # Align Camarilla levels to original timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R1 with uptrend and volume spike
            long_signal = (close[i] > R1_1d_aligned[i]) and (close[i] > ema_50_1w_aligned[i]) and vol_spike[i]
            # Short: price breaks below S1 with downtrend and volume spike
            short_signal = (close[i] < S1_1d_aligned[i]) and (close[i] < ema_50_1w_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: price touches S1 or trend reverses
            exit_signal = (close[i] < S1_1d_aligned[i]) or (close[i] < ema_50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price touches R1 or trend reverses
            exit_signal = (close[i] > R1_1d_aligned[i]) or (close[i] > ema_50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0