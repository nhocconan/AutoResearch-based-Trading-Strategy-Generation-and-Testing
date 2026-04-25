#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm
Hypothesis: 4h Camarilla R1/S1 breakout with 1d trend filter (price > 1d EMA34) and volume confirmation.
Goes long when price breaks above R1 with 1d uptrend and volume spike.
Short when price breaks below S1 with 1d downtrend and volume spike.
Exit when price reverts to the 1d EMA34 or opposite Camarilla level is touched.
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
    
    # Get 4h data for Camarilla calculations (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla parameters
    lookback = 20
    
    # Calculate Camarilla levels for each 4h bar
    R1_4h = np.full(len(close_4h), np.nan)
    S1_4h = np.full(len(close_4h), np.nan)
    PP_4h = np.full(len(close_4h), np.nan)
    
    for i in range(lookback, len(close_4h)):
        # Use the last 20 4h bars including current
        high_max = np.max(high_4h[i-lookback+1:i+1])
        low_min = np.min(low_4h[i-lookback+1:i+1])
        close_val = close_4h[i]
        
        # Pivot point
        PP = (high_max + low_min + close_val) / 3
        # Camarilla R1 and S1
        R1 = close_val + (1.1/12) * (high_max - low_min)
        S1 = close_val - (1.1/12) * (high_max - low_min)
        
        PP_4h[i] = PP
        R1_4h[i] = R1
        S1_4h[i] = S1
    
    # Align Camarilla levels to original timeframe
    PP_4h_aligned = align_htf_to_ltf(prices, df_4h, PP_4h)
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
            np.isnan(PP_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R1, 1d uptrend, volume spike
            long_signal = (close[i] > R1_4h_aligned[i]) and (close[i] > ema_34_1d_aligned[i]) and vol_spike[i]
            # Short: price breaks below S1, 1d downtrend, volume spike
            short_signal = (close[i] < S1_4h_aligned[i]) and (close[i] < ema_34_1d_aligned[i]) and vol_spike[i]
            
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
            # Exit when price reverts to 1d EMA34 or touches S1 (mean reversion)
            exit_signal = (close[i] <= ema_34_1d_aligned[i]) or (close[i] <= S1_4h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price reverts to 1d EMA34 or touches R1 (mean reversion)
            exit_signal = (close[i] >= ema_34_1d_aligned[i]) or (close[i] >= R1_4h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0