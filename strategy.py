#!/usr/bin/env python3
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
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    r4 = r3 + (high_1d - low_1d)
    s4 = s3 - (high_1d - low_1d)
    
    # Use previous day's pivots (avoid look-ahead)
    r4_prev = np.roll(r4, 1)
    s4_prev = np.roll(s4, 1)
    r4_prev[0] = np.nan
    s4_prev[0] = np.nan
    
    # Align weekly pivot levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_prev)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_prev)
    
    # Volume confirmation: current volume > 1.5 * 4-period average (6h * 4 = 24h)
    volume_ma4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need R4/S4 and ATR MA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma4[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(r4_6h[i]) or 
            np.isnan(s4_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 4-period average
        volume_filter = volume[i] > (1.5 * volume_ma4[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        
        if position == 0:
            # Long: price breaks above R4 with volume and volatility
            if close[i] > r4_6h[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume and volatility
            elif close[i] < s4_6h[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R3 or volatility drops
            # Need R3 for exit condition
            r3 = pivot[i] + 2 * (high_1d[i] - low_1d[i])  # Recalculate for current day
            r3_6h = align_htf_to_ltf(prices, df_1d, np.roll(r3, 1))[i] if i > 0 else np.nan
            if not np.isnan(r3_6h) and close[i] < r3_6h:
                signals[i] = 0.0
                position = 0
            elif not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S3 or volatility drops
            # Need S3 for exit condition
            s3 = pivot[i] - 2 * (high_1d[i] - low_1d[i])  # Recalculate for current day
            s3_6h = align_htf_to_ltf(prices, df_1d, np.roll(s3, 1))[i] if i > 0 else np.nan
            if not np.isnan(s3_6h) and close[i] > s3_6h:
                signals[i] = 0.0
                position = 0
            elif not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R4_S4_Breakout_Volume"
timeframe = "6h"
leverage = 1.0