#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_surge_v1"
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
    
    # Get daily data for TRIX and pivot calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Calculate TRIX on daily close
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False).mean().values
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # Apply alignment with 1-bar delay for completed daily bar
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_raw)
    
    # Calculate daily pivot points (standard: (H+L+C)/3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        pivot[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume surge filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # warmup for volume MA
        # Skip if not ready
        if np.isnan(trix_aligned[i]) or np.isnan(pivot_aligned[i]) or np.isnan(volume_surge[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        long_signal = trix_aligned[i] > 0 and close[i] > pivot_aligned[i] and volume_surge[i]
        short_signal = trix_aligned[i] < 0 and close[i] < pivot_aligned[i] and volume_surge[i]
        
        # Exit when TRIX changes sign or price crosses pivot in opposite direction
        exit_long = trix_aligned[i] < 0 or close[i] < pivot_aligned[i]
        exit_short = trix_aligned[i] > 0 or close[i] > pivot_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals