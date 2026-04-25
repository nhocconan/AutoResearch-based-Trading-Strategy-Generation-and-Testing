#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dEMA200_Trend_VolumeSpike
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA200 trend filter and volume spike confirmation.
Long when price breaks above R3 with 1d EMA200 uptrend and volume > 2.0x 20-period average.
Short when price breaks below S3 with 1d EMA200 downtrend and volume > 2.0x 20-period average.
Exit on opposite band touch (R1/S1) or trend reversal.
Uses discrete sizing (0.25) to balance return and drawdown. Target: 12-37 trades/year.
Works in bull via trend-following breakouts, in bear via mean reversion at extreme bands.
12h timeframe reduces trade frequency to avoid fee drag while capturing multi-day moves.
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
    
    # Get 12h data for Camarilla calculations (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar (based on previous bar)
    R1_12h = np.full(len(close_12h), np.nan)
    S1_12h = np.full(len(close_12h), np.nan)
    R3_12h = np.full(len(close_12h), np.nan)
    S3_12h = np.full(len(close_12h), np.nan)
    R4_12h = np.full(len(close_12h), np.nan)
    S4_12h = np.full(len(close_12h), np.nan)
    
    for i in range(1, len(close_12h)):
        # Camarilla levels based on previous 12h bar's range
        high_prev = high_12h[i-1]
        low_prev = low_12h[i-1]
        close_prev = close_12h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R1_12h[i] = close_prev + (range_prev * 1.1 / 12)
            S1_12h[i] = close_prev - (range_prev * 1.1 / 12)
            R3_12h[i] = close_prev + (range_prev * 1.1 / 4)
            S3_12h[i] = close_prev - (range_prev * 1.1 / 4)
            R4_12h[i] = close_prev + (range_prev * 1.1 / 2)
            S4_12h[i] = close_prev - (range_prev * 1.1 / 2)
    
    # Align Camarilla levels to original timeframe
    R1_12h_aligned = align_htf_to_ltf(prices, df_12h, R1_12h)
    S1_12h_aligned = align_htf_to_ltf(prices, df_12h, S1_12h)
    R3_12h_aligned = align_htf_to_ltf(prices, df_12h, R3_12h)
    S3_12h_aligned = align_htf_to_ltf(prices, df_12h, S3_12h)
    R4_12h_aligned = align_htf_to_ltf(prices, df_12h, R4_12h)
    S4_12h_aligned = align_htf_to_ltf(prices, df_12h, S4_12h)
    
    # Get 1d data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA200 for trend
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_12h_aligned[i]) or np.isnan(S3_12h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R3 with uptrend and volume spike
            long_signal = (close[i] > R3_12h_aligned[i]) and (close[i] > ema_200_1d_aligned[i]) and vol_spike[i]
            # Short: price breaks below S3 with downtrend and volume spike
            short_signal = (close[i] < S3_12h_aligned[i]) and (close[i] < ema_200_1d_aligned[i]) and vol_spike[i]
            
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
            exit_signal = (close[i] < S1_12h_aligned[i]) or (close[i] < ema_200_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price touches R1 or trend reverses
            exit_signal = (close[i] > R1_12h_aligned[i]) or (close[i] > ema_200_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA200_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0