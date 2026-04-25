#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm
Hypothesis: Daily Camarilla R1/S1 breakout with weekly trend filter (price > weekly EMA50) and volume confirmation.
Goes long when price breaks above R1 with weekly uptrend and volume spike.
Short when price breaks below S1 with weekly downtrend and volume spike.
Exit when price reverts to the weekly EMA50 or opposite Camarilla level is touched.
Uses discrete sizing (0.25) to minimize fees. Target: 20-50 trades/year.
Works in bull via breakouts with trend, in bear via mean reversion at Camarilla levels.
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla parameters
    lookback = 20
    
    # Calculate Camarilla levels for each 1d bar
    R1_1d = np.full(len(close_1d), np.nan)
    S1_1d = np.full(len(close_1d), np.nan)
    PP_1d = np.full(len(close_1d), np.nan)
    
    for i in range(lookback, len(close_1d)):
        # Use the last 20 1d bars including current
        high_max = np.max(high_1d[i-lookback+1:i+1])
        low_min = np.min(low_1d[i-lookback+1:i+1])
        close_val = close_1d[i]
        
        # Pivot point
        PP = (high_max + low_min + close_val) / 3
        # Camarilla R1 and S1
        R1 = close_val + (1.1/12) * (high_max - low_min)
        S1 = close_val - (1.1/12) * (high_max - low_min)
        
        PP_1d[i] = PP
        R1_1d[i] = R1
        S1_1d[i] = S1
    
    # Align Camarilla levels to original timeframe
    PP_1d_aligned = align_htf_to_ltf(prices, df_1d, PP_1d)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Get 1w data for trend filter
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
            np.isnan(PP_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R1, weekly uptrend, volume spike
            long_signal = (close[i] > R1_1d_aligned[i]) and (close[i] > ema_50_1w_aligned[i]) and vol_spike[i]
            # Short: price breaks below S1, weekly downtrend, volume spike
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
            # Exit when price reverts to weekly EMA50 or touches S1 (mean reversion)
            exit_signal = (close[i] <= ema_50_1w_aligned[i]) or (close[i] <= S1_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price reverts to weekly EMA50 or touches R1 (mean reversion)
            exit_signal = (close[i] >= ema_50_1w_aligned[i]) or (close[i] >= R1_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0