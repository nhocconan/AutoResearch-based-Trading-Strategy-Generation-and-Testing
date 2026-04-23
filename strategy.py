#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
- Long: Close breaks above Camarilla R1 (1d) + volume > 1.5x 20-period average + price > 12h EMA50
- Short: Close breaks below Camarilla S1 (1d) + volume > 1.5x 20-period average + price < 12h EMA50
- Exit: Close breaks below Camarilla S3 for long OR above Camarilla R3 for short OR 12h EMA50 trend flip
- Uses Camarilla pivot levels for institutional support/resistance, volume for conviction, 12h EMA50 for HTF trend
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (breakouts in uptrend) and bear (breakdowns in downtrend) markets
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d OHLC for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels: R1, S1, R3, S3
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # R3 = close + 1.1*(high-low)/4
    # S3 = close - 1.1*(high-low)/4
    rang = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * rang / 12
    camarilla_s1 = close_1d - 1.1 * rang / 12
    camarilla_r3 = close_1d + 1.1 * rang / 4
    camarilla_s3 = close_1d - 1.1 * rang / 4
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Camarilla R1 + volume confirmation + price > 12h EMA50
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_confirm and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S1 + volume confirmation + price < 12h EMA50
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Camarilla S3 OR price < 12h EMA50 (trend flip)
            if (close[i] < camarilla_s3_aligned[i]) or (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Camarilla R3 OR price > 12h EMA50 (trend flip)
            if (close[i] > camarilla_r3_aligned[i]) or (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0