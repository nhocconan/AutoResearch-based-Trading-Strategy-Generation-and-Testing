#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R4_S4_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and pivot levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_w = pd.Series(df_w['close'].values)
    ema50_w = close_w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)
    
    # Calculate weekly pivot levels from previous week OHLC
    H = df_w['high'].values
    L = df_w['low'].values
    C = df_w['close'].values
    
    # Pivot point: P = (H + L + C) / 3
    P = (H + L + C) / 3
    
    # Weekly R4 and S4 levels (breakout levels)
    # R4 = C + (H - L) * 1.1
    # S4 = C - (H - L) * 1.1
    R4 = C + (H - L) * 1.1
    S4 = C - (H - L) * 1.1
    
    # Align weekly levels to 6h timeframe (use previous week's levels)
    R4_aligned = align_htf_to_ltf(prices, df_w, R4)
    S4_aligned = align_htf_to_ltf(prices, df_w, S4)
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_series = pd.Series(volume)
    vol_ma24 = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_w_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma24[i]
        
        if position == 0:
            # Long: Price breaks above R4 with volume and above weekly EMA trend
            if close[i] > R4_aligned[i] and vol_ok and close[i] > ema50_w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S4 with volume and below weekly EMA trend
            elif close[i] < S4_aligned[i] and vol_ok and close[i] < ema50_w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back below S4 (trend reversal)
            if close[i] < S4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses back above R4
            if close[i] > R4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals