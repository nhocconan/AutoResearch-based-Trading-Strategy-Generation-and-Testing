#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1wTrend_Volume
Hypothesis: Use Camarilla R3/S3 levels from daily data as breakout signals, with weekly trend filter (price > weekly SMA50) and volume confirmation (volume > 1.5x average volume). Goes long on breakout above R3, short on breakdown below S3. Camarilla levels work well in ranging markets (2025) and capture breakouts in trends. Weekly SMA50 filter avoids counter-trend trades. Volume confirmation reduces false breakouts. Designed for 4h timeframe to target 25-40 trades/year.
"""

name = "4h_Camarilla_R3S3_Breakout_1wTrend_Volume"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # R4 = Close + 1.5*(High-Low), R3 = Close + 1.1*(High-Low), S3 = Close - 1.1*(High-Low), S4 = Close - 1.5*(High-Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (no extra delay for price levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get weekly SMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    sma_50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate average volume (20-period) for volume confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Breakout above R3 with volume confirmation and uptrend
            if (close[i] > r3_aligned[i] and 
                volume[i] > 1.5 * vol_avg[i] and 
                close[i] > sma_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3 with volume confirmation and downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume[i] > 1.5 * vol_avg[i] and 
                  close[i] < sma_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or trend turns down
            if close[i] < s3_aligned[i] or close[i] < sma_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or trend turns up
            if close[i] > r3_aligned[i] or close[i] > sma_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals