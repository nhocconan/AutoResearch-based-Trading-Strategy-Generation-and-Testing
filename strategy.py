#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_Volume
Hypothesis: Use daily Camarilla pivot levels (R3/S3) for breakout signals with volume confirmation and EMA34 trend filter. 
Long when price breaks above R3 with volume > 1.5x average and close > EMA34. 
Short when price breaks below S3 with volume > 1.5x average and close < EMA34.
Exit on opposite Camarilla level (R1/S1) break. Designed for 4h to capture swings with low frequency.
Works in both bull and bear markets due to directional breakout logic and trend filter.
"""

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R3 = pivot + (range_1d * 1.1 / 2)
    R1 = pivot + (range_1d * 1.1 / 4)
    S1 = pivot - (range_1d * 1.1 / 4)
    S3 = pivot - (range_1d * 1.1 / 2)
    
    # Calculate daily EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h timeframe with proper delay for daily data
    # Camarilla levels need 1 bar delay (previous day's levels)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3, additional_delay_bars=1)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1, additional_delay_bars=1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1, additional_delay_bars=1)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3, additional_delay_bars=1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d, additional_delay_bars=1)
    
    # Volume confirmation: 20-period average on 4h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for daily EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation and uptrend
            if (close[i] > R3_aligned[i] and 
                vol_ratio[i] > 1.5 and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume confirmation and downtrend
            elif (close[i] < S3_aligned[i] and 
                  vol_ratio[i] > 1.5 and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below R1 (take profit) or S1 (stop)
            if close[i] < R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above S1 (take profit) or R1 (stop)
            if close[i] > S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals