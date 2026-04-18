#!/usr/bin/env python3
"""
6h_1W_1D_ThreeBarBreakout_Volume
Hypothesis: Three consecutive closes above/below weekly pivot + daily EMA34 for trend, with volume > 1.5x 20-period average.
Targets breakouts in trending markets (both bull/bear) with momentum confirmation. Uses weekly pivot for structure,
daily EMA for direction, and volume for confirmation. Designed for 15-25 trades/year on 6H to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_to_ltf, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    # Get daily data for EMA and pivot
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly pivot (using prior week's OHLC)
    prev_close_1w = np.roll(df_1w['close'].values, 1)
    prev_high_1w = np.roll(df_1w['high'].values, 1)
    prev_low_1w = np.roll(df_1w['low'].values, 1)
    prev_close_1w[0] = df_1w['close'].values[0]
    prev_high_1w[0] = df_1w['high'].values[0]
    prev_low_1w[0] = df_1w['low'].values[0]
    
    # Classic pivot point
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    # R1 and S1 levels
    r1_1w = 2 * pivot_1w - prev_low_1w
    s1_1w = 2 * pivot_1w - prev_high_1w
    
    # Daily EMA34 for trend
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly and daily data to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: three consecutive closes above weekly pivot AND above daily EMA34 with volume
            if (close[i-2] > pivot_1w_aligned[i-2] and close[i-1] > pivot_1w_aligned[i-1] and 
                close[i] > pivot_1w_aligned[i] and close[i-2] > ema_34_aligned[i-2] and 
                close[i-1] > ema_34_aligned[i-1] and close[i] > ema_34_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: three consecutive closes below weekly pivot AND below daily EMA34 with volume
            elif (close[i-2] < pivot_1w_aligned[i-2] and close[i-1] < pivot_1w_aligned[i-1] and 
                  close[i] < pivot_1w_aligned[i] and close[i-2] < ema_34_aligned[i-2] and 
                  close[i-1] < ema_34_aligned[i-1] and close[i] < ema_34_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below weekly pivot OR below daily EMA34
            if close[i] < pivot_1w_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly pivot OR above daily EMA34
            if close[i] > pivot_1w_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1W_1D_ThreeBarBreakout_Volume"
timeframe = "6h"
leverage = 1.0