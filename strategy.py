#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_MACD_Histogram_Trend_With_1d_Volume"
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
    
    # Get daily data for MACD and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # MACD on daily: fast EMA(12), slow EMA(26), signal EMA(9)
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Volume SMA(20) on daily
    vol_sma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align MACD histogram and volume SMA to 12h
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)
    vol_sma_aligned = align_htf_to_ltf(prices, df_1d, vol_sma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # warmup for MACD
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(macd_hist_aligned[i]) or np.isnan(vol_sma_aligned[i]) or 
            np.isnan(volume_1d[i // 288])):  # volume index for current 12h bar
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current 12h volume (from original 12h data)
        vol_12h = volume[i]
        # Daily volume SMA from aligned array (already shifted for completed bar)
        vol_sma_val = vol_sma_aligned[i]
        
        if position == 0:
            # Long: MACD histogram positive and rising + volume above average
            macd_rising = (i > start_idx and macd_hist_aligned[i] > macd_hist_aligned[i-1])
            long_cond = (macd_hist_aligned[i] > 0) and macd_rising and (vol_12h > vol_sma_val)
            
            # Short: MACD histogram negative and falling + volume above average
            macd_falling = (i > start_idx and macd_hist_aligned[i] < macd_hist_aligned[i-1])
            short_cond = (macd_hist_aligned[i] < 0) and macd_falling and (vol_12h > vol_sma_val)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: MACD histogram turns negative
            if macd_hist_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: MACD histogram turns positive
            if macd_hist_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: MACD histogram on daily timeframe captures momentum shifts, confirmed by volume expansion.
# Long when MACD histogram is positive AND rising (bullish momentum building) with above-average volume.
# Short when MACD histogram is negative AND falling (bearish momentum building) with above-average volume.
# Exits when histogram crosses zero, indicating momentum reversal.
# Uses 12h timeframe for execution to reduce trade frequency (target: 50-150 total trades over 4 years).
# Volume confirmation ensures moves are supported by participation, reducing false signals.
# Works in both bull and bear markets by following momentum shifts.