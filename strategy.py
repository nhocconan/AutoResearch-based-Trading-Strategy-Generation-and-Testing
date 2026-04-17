#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Camarilla R1/S1 breakout + volume confirmation + ATR volatility filter.
Long when price breaks above 1d Camarilla R1 with volume > 1.3x 20-period average and current ATR < 1.5x 20-period ATR average.
Short when price breaks below 1d Camarilla S1 with volume > 1.3x 20-period average and current ATR < 1.5x 20-period ATR average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
Volatility filter ensures breakouts occur during consolidation, reducing false signals in choppy markets.
Works in bull markets (trend continuation) and bear markets (mean reversion after low volatility periods).
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
    
    # Get daily data for Camarilla levels, volume, and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Camarilla levels (based on previous day's range)
    # R1 = Close + 1.1*(High-Low)/2
    # S1 = Close - 1.1*(High-Low)/2
    range_1d = high_1d - low_1d
    r1 = close_1d + 1.1 * range_1d / 2
    s1 = close_1d - 1.1 * range_1d / 2
    
    # Calculate daily ATR (14-period) for volatility filter
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ATR and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(atr_ma_20_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_1d_aligned[i]
        
        # Volatility filter: current ATR < 1.5x 20-period ATR average (breakout from low volatility)
        vol_filter = atr_aligned[i] < 1.5 * atr_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above daily Camarilla R1 with volume and low volatility
            if (close[i] > r1_aligned[i] and 
                volume_confirmed and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Camarilla S1 with volume and low volatility
            elif (close[i] < s1_aligned[i] and 
                  volume_confirmed and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below daily Camarilla midpoint (S1+R1)/2
            midpoint = (r1 + s1) / 2
            midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint)
            if close[i] < midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above daily Camarilla midpoint (S1+R1)/2
            midpoint = (r1 + s1) / 2
            midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint)
            if close[i] > midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dCamarilla_R1S1_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0