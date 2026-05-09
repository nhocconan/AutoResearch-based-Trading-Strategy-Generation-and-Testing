#!/usr/bin/env python3
# 4H_12H_Camarilla_R2_S2_Breakout_12hEMA50_Trend_Volume
# Hypothesis: Combine 4h breakouts at 12h Camarilla R2/S2 levels with 12h EMA50 trend filter and volume confirmation.
# The 12h timeframe filters noise while capturing major swings. R2/S2 levels provide stronger breakout signals than R1/S1.
# EMA50 on 12h offers robust trend filtering suitable for swing trading. Volume ensures breakouts have conviction.
# Designed to work in both bull and bear markets via trend filter. Target: 20-50 trades/year per symbol.

name = "4H_12H_Camarilla_R2_S2_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for Camarilla pivot levels and EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point and Camarilla levels (R2, S2)
    pivot = (high_12h + low_12h + close_12h) / 3
    range_ = high_12h - low_12h
    r2 = pivot + range_ * 1.1 / 2  # R2 = pivot + (range * 1.1 / 2)
    s2 = pivot - range_ * 1.1 / 2  # S2 = pivot - (range * 1.1 / 2)
    
    # Get 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R2 + above 12h EMA50 + volume confirmation
            if close[i] > r2_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S2 + below 12h EMA50 + volume confirmation
            elif close[i] < s2_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 12h EMA50 (trend change)
            if close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 12h EMA50 (trend change)
            if close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals