#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1wPivot_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points and R1/S1 levels
    H_w = df_w['high'].values
    L_w = df_w['low'].values
    C_w = df_w['close'].values
    
    # Pivot point: P = (H + L + C) / 3
    P_w = (H_w + L_w + C_w) / 3
    
    # Weekly R1 and S1 levels
    # R1 = 2*P - L
    # S1 = 2*P - H
    R1_w = 2 * P_w - L_w
    S1_w = 2 * P_w - H_w
    
    # Align weekly levels to 12h timeframe (use previous week's levels)
    R1_w_aligned = align_htf_to_ltf(prices, df_w, R1_w)
    S1_w_aligned = align_htf_to_ltf(prices, df_w, S1_w)
    
    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    close_d = pd.Series(df_d['close'].values)
    ema34_d = close_d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_series = pd.Series(volume)
    vol_ma24 = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_w_aligned[i]) or np.isnan(S1_w_aligned[i]) or 
            np.isnan(ema34_d_aligned[i]) or np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma24[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume and above daily EMA trend
            if close[i] > R1_w_aligned[i] and vol_ok and close[i] > ema34_d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and below daily EMA trend
            elif close[i] < S1_w_aligned[i] and vol_ok and close[i] < ema34_d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back below S1 (trend reversal)
            if close[i] < S1_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses back above R1
            if close[i] > R1_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals