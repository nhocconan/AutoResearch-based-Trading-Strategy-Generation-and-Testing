#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 level AND close > 1w EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S1 level AND close < 1w EMA50 AND volume > 2.0x 20-period average.
Exit when price crosses Camarilla Pivot point (central level).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 7-25 trades/year per symbol.
1d Camarilla levels derived from prior day's OHLC provide meaningful breakout thresholds.
1w EMA50 offers smooth HTF trend filter with lower lag than longer EMAs. Volume confirmation ensures institutional participation.
Designed to work in both bull and bear markets by using HTF trend filter and volatility-adjusted entries.
"""

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
    
    # Load 1d data for Camarilla levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d OHLC
    range_1d = high_1d - low_1d
    camarilla_r1_1d = close_1d + 1.083 * range_1d   # R1: close + 1.083 * range
    camarilla_s1_1d = close_1d - 1.083 * range_1d   # S1: close - 1.083 * range
    camarilla_pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Ensure warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND close > 1w EMA50 AND volume spike
            if (price > camarilla_r1_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 AND close < 1w EMA50 AND volume spike
            elif (price < camarilla_s1_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Camarilla Pivot point
            if position == 1 and price < camarilla_pivot_aligned[i]:
                exit_signal = True
            elif position == -1 and price > camarilla_pivot_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Camarilla_R1S1_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0