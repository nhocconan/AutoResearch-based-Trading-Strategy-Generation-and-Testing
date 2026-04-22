#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla pivot reversal with 12-hour trend filter and volume confirmation.
Long when price touches S1 support in uptrend with above-average volume.
Short when price touches R1 resistance in downtrend with above-average volume.
Exit when price reaches opposite Camarilla level (S3 for long, R3 for short) or trend reverses.
Camarilla levels provide precise intraday support/resistance; trend filter ensures directional alignment;
volume filter confirms institutional interest. Works in both bull and bear markets by combining
mean-reversion at pivot levels with trend-following filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Camarilla levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: Range = (high - low)
    # Resistance levels: R1 = close + (range * 1.1/12), R2 = close + (range * 1.1/6), R3 = close + (range * 1.1/4)
    # Support levels: S1 = close - (range * 1.1/12), S2 = close - (range * 1.1/6), S3 = close - (range * 1.1/4)
    range_1d = high_1d - low_1d
    R1 = close_1d + (range_1d * 1.1 / 12)
    R3 = close_1d + (range_1d * 1.1 / 4)
    S1 = close_1d - (range_1d * 1.1 / 12)
    S3 = close_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA50 for trend identification
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation (4h volume > 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price at S1 support, uptrend (price > EMA50), volume confirmation
            if (abs(close[i] - S1_aligned[i]) / S1_aligned[i] < 0.005 and  # Within 0.5% of S1
                close[i] > ema_50_12h_aligned[i] and  # Uptrend filter
                volume[i] > vol_ma_20[i]):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price at R1 resistance, downtrend (price < EMA50), volume confirmation
            elif (abs(close[i] - R1_aligned[i]) / R1_aligned[i] < 0.005 and  # Within 0.5% of R1
                  close[i] < ema_50_12h_aligned[i] and  # Downtrend filter
                  volume[i] > vol_ma_20[i]):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price reaches S3 (strong support break) or trend turns down
                if close[i] <= S3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price reaches R3 (strong resistance break) or trend turns up
                if close[i] >= R3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Reversal_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0