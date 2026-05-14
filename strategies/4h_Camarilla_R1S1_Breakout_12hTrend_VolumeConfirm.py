#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrend_VolumeConfirm
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume spike confirmation.
Long when price breaks above R1 with 12h EMA34 uptrend and volume > 2.0x 20-period average.
Short when price breaks below S1 with 12h EMA34 downtrend and volume > 2.0x 20-period average.
Exit on opposite band touch (S1/R1) or trend reversal.
Uses discrete sizing (0.25) to minimize fee churn. Target: 20-50 trades/year.
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
    
    # Get 4h data for Camarilla calculations (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar (based on previous bar)
    R1_4h = np.full(len(close_4h), np.nan)
    S1_4h = np.full(len(close_4h), np.nan)
    R3_4h = np.full(len(close_4h), np.nan)
    S3_4h = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        # Camarilla levels based on previous 4h bar's range
        high_prev = high_4h[i-1]
        low_prev = low_4h[i-1]
        close_prev = close_4h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R1_4h[i] = close_prev + (range_prev * 1.1 / 12)
            S1_4h[i] = close_prev - (range_prev * 1.1 / 12)
            R3_4h[i] = close_prev + (range_prev * 1.1 / 4)
            S3_4h[i] = close_prev - (range_prev * 1.1 / 4)
    
    # Align Camarilla levels to original timeframe
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    R3_4h_aligned = align_htf_to_ltf(prices, df_4h, R3_4h)
    S3_4h_aligned = align_htf_to_ltf(prices, df_4h, S3_4h)
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA34 for trend
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R1 with uptrend and volume spike
            long_signal = (close[i] > R1_4h_aligned[i]) and (close[i] > ema_34_12h_aligned[i]) and vol_spike[i]
            # Short: price breaks below S1 with downtrend and volume spike
            short_signal = (close[i] < S1_4h_aligned[i]) and (close[i] < ema_34_12h_aligned[i]) and vol_spike[i]
            
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
            exit_signal = (close[i] < S1_4h_aligned[i]) or (close[i] < ema_34_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price touches R1 or trend reverses
            exit_signal = (close[i] > R1_4h_aligned[i]) or (close[i] > ema_34_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0