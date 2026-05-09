#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_2"
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
    
    # Get daily data for trend filter and pivot levels
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    close_d = pd.Series(df_d['close'].values)
    ema34_d = close_d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    
    # Calculate daily pivot levels from previous day OHLC
    H = df_d['high'].values
    L = df_d['low'].values
    C = df_d['close'].values
    
    # Pivot point: P = (H + L + C) / 3
    P = (H + L + C) / 3
    
    # Daily R1 and S1 levels (breakout levels)
    # R1 = (2 * P) - L
    # S1 = (2 * P) - H
    R1 = (2 * P) - L
    S1 = (2 * P) - H
    
    # Align daily levels to 4h timeframe (use previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_d, S1)
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume and above daily EMA trend
            if close[i] > R1_aligned[i] and vol_ok and close[i] > ema34_d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and below daily EMA trend
            elif close[i] < S1_aligned[i] and vol_ok and close[i] < ema34_d_aligned[i]:
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