#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Camarilla pivot levels (S3/R3) with volume confirmation and 1-day trend filter.
# Long when price > 1-day EMA34 (bullish trend), price breaks above S3 level, volume > 1.5x average.
# Short when price < 1-day EMA34 (bearish trend), price breaks below R3 level, volume > 1.5x average.
# Exit on trend reversal or price crosses opposite pivot level (S1/R1).
# Uses position size 0.25 to balance return and drawdown. Target: 20-50 trades/year per symbol.
# Designed to capture breakouts in both bull and bear markets using 1-day trend filter, with volume to confirm breakout strength.

name = "4h_1dEMA34_Camarilla_S3R3_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1-day EMA(34)
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Previous day's Camarilla levels
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # Avoid NaN on first element
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Calculate Camarilla levels: S1, S2, S3, R1, R2, R3
    range_1d = prev_high_1d - prev_low_1d
    S1 = prev_close_1d - range_1d * 1.0 / 12
    S2 = prev_close_1d - range_1d * 2.0 / 12
    S3 = prev_close_1d - range_1d * 3.0 / 12
    R1 = prev_close_1d + range_1d * 1.0 / 12
    R2 = prev_close_1d + range_1d * 2.0 / 12
    R3 = prev_close_1d + range_1d * 3.0 / 12
    
    # Align Camarilla levels to 4h timeframe
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d EMA bullish (price > EMA), price breaks above S3 level, volume spike
            if (close[i] > ema_1d_aligned[i] and
                close[i] > S3_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: 1d EMA bearish (price < EMA), price breaks below R3 level, volume spike
            elif (close[i] < ema_1d_aligned[i] and
                  close[i] < R3_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend reversal or price crosses below S1 level
            if (close[i] < ema_1d_aligned[i] or 
                close[i] < S1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal or price crosses above R1 level
            if (close[i] > ema_1d_aligned[i] or 
                close[i] > R1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals