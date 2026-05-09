#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1wPivot_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (EMA200)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA200 for trend filter
    close_d = pd.Series(df_d['close'].values)
    ema200_d = close_d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_d_aligned = align_htf_to_ltf(prices, df_d, ema200_d)
    
    # Get weekly data for pivot levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot levels from previous week OHLC
    H = df_w['high'].values
    L = df_w['low'].values
    C = df_w['close'].values
    
    # Pivot point: P = (H + L + C) / 3
    P = (H + L + C) / 3
    
    # Weekly R1 and S1 levels (standard pivot levels)
    # R1 = 2*P - L
    # S1 = 2*P - H
    R1 = 2 * P - L
    S1 = 2 * P - H
    
    # Align weekly levels to 12h timeframe (use previous week's levels)
    R1_aligned = align_htf_to_ltf(prices, df_w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_w, S1)
    ema200_d_aligned = align_htf_to_ltf(prices, df_d, ema200_d)
    
    # Volume confirmation: current volume > 1.5x 24-period average (adjusted for 12h)
    vol_series = pd.Series(volume)
    vol_ma24 = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema200_d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma24[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume and above daily EMA200 trend
            if close[i] > R1_aligned[i] and vol_ok and close[i] > ema200_d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and below daily EMA200 trend
            elif close[i] < S1_aligned[i] and vol_ok and close[i] < ema200_d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back below S1 (trend reversal)
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses back above R1
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals