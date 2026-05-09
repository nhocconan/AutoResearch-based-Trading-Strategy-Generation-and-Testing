#!/usr/bin/env python3
# 4H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Strategy: Breakout of Camarilla R3/S3 levels with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above R3, above 1d EMA34, and volume > 2x 20-period average
# Short when price breaks below S3, below 1d EMA34, and volume > 2x 20-period average
# Exit when price crosses back below/above the respective Camarilla level OR trend reverses
# Position size: 0.25 (25% of capital) to balance return and drawdown
# Designed for 4h timeframe with daily trend filter to reduce whipsaw and target 20-50 trades/year

name = "4H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d EMA34 for trend filter
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily pivot and Camarilla levels
    # Pivot = (H + L + C) / 3
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # R3 = C + (H - L) * 1.1/2
    # S3 = C - (H - L) * 1.1/2
    range_hl = df_1d['high'] - df_1d['low']
    R3 = df_1d['close'] + range_hl * 1.1 / 2
    S3 = df_1d['close'] - range_hl * 1.1 / 2
    
    # Align 1d Camarilla levels to 4h timeframe (waits for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3, above EMA34 (bullish trend) + volume spike
            if (close[i] > R3_aligned[i] and 
                close[i] > ema34_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, below EMA34 (bearish trend) + volume spike
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema34_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below R3 OR trend turns bearish
            if (close[i] < R3_aligned[i]) or (close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above S3 OR trend turns bullish
            if (close[i] > S3_aligned[i]) or (close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals