#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_trix_volume_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for TRIX calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate TRIX(18) on 4h data: triple EMA of % change
    # Step 1: EMA1 of close
    ema1 = pd.Series(close_4h).ewm(span=18, adjust=False, min_periods=18).mean().values
    # Step 2: EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=18, adjust=False, min_periods=18).mean().values
    # Step 3: EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=18, adjust=False, min_periods=18).mean().values
    # Step 4: Calculate % change of EMA3
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # Smooth TRIX with signal line (9-period EMA)
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX signal to 1h timeframe
    trix_signal_aligned = align_htf_to_ltf(prices, df_4h, trix_signal)
    
    # Get 1d data for volume filter (average volume)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Calculate 20-day average volume on daily data
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume filter: current 1h volume > 1.5x 20-day average volume (scaled to hourly)
    # Approximate: 1 day = 24 hours, so scale daily avg volume to hourly expectation
    vol_threshold = vol_ma_1d_aligned * (24/1.5)  # Adjust for timeframe difference
    volume_ok = volume > vol_threshold
    
    # Session filter: 08-20 UTC (avoid low liquidity periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if not ready or outside session
        if (np.isnan(trix_signal_aligned[i]) or 
            np.isnan(volume_ok[i]) or 
            not session_ok[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # TRIX signals: cross above/below zero line
        trix_value = trix_signal_aligned[i]
        trix_prev = trix_signal_aligned[i-1] if i > 0 else 0
        
        # Long: TRIX crosses above zero with volume confirmation
        long_signal = (trix_value > 0 and trix_prev <= 0) and volume_ok[i]
        # Short: TRIX crosses below zero with volume confirmation
        short_signal = (trix_value < 0 and trix_prev >= 0) and volume_ok[i]
        
        # Exit when TRIX reverses cross
        exit_long = (trix_value < 0 and trix_prev >= 0)
        exit_short = (trix_value > 0 and trix_prev <= 0)
        
        # Execute trades with fixed size 0.20
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals