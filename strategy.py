#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla R1/S1 breakout with 12-hour EMA(50) trend filter and volume spike confirmation.
Trades breakouts in the direction of the 12h trend only when volume exceeds 2.0x the 20-period average.
Uses fixed position size of 0.25 to balance risk and reward. Targets 20-40 trades/year (80-160 total over 4 years)
to minimize fee drag while capturing significant moves in both bull and bear markets via trend alignment.
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
    
    # Load 4h data for Camarilla calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # We need daily OHLC for Camarilla, so we'll use 1d data shifted to align with 4h periods
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1, R2, S2, R3, S3, R4, S4
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # R2 = close + (high - low) * 1.1/6
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # S2 = close - (high - low) * 1.1/6
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    
    range_1d = high_1d - low_1d
    r4 = close_1d + range_1d * 1.1 / 2
    r3 = close_1d + range_1d * 1.1 / 4
    r2 = close_1d + range_1d * 1.1 / 6
    r1 = close_1d + range_1d * 1.1 / 12
    s1 = close_1d - range_1d * 1.1 / 12
    s2 = close_1d - range_1d * 1.1 / 6
    s3 = close_1d - range_1d * 1.1 / 4
    s4 = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels for current 4h bars)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA for trend filter (50-period)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above R1, above 12h EMA (uptrend)
            if close[i] > r1_aligned[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below 12h EMA (downtrend)
            elif close[i] < s1_aligned[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches S1 or closes below 12h EMA
                if close[i] < s1_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches R1 or closes above 12h EMA
                if close[i] > r1_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0