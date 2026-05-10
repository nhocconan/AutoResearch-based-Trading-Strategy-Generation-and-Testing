#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Uses Camarilla pivot levels (R1/S1) from 1d timeframe for entry, with 12h EMA50 for trend direction and volume confirmation.
# Works in both bull and bear markets by following the 12h trend: long when price breaks above R1 in uptrend, short when breaks below S1 in downtrend.
# Volume filter ensures breakout validity. Designed for low trade frequency (20-40 trades/year) to minimize fee drag.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    R1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    S1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 12h EMA50 for trend direction
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, uptrend (price > EMA50), volume confirmation
            if (close[i] > R1_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, downtrend (price < EMA50), volume confirmation
            elif (close[i] < S1_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below EMA50 (trend reversal) OR price drops below S1 (failed breakout)
            if (close[i] < ema_50_aligned[i] or 
                close[i] < S1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above EMA50 (trend reversal) OR price rises above R1 (failed breakout)
            if (close[i] > ema_50_aligned[i] or 
                close[i] > R1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals