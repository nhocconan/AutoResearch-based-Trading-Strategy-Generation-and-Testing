#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Williams %R + volume confirmation + ATR-based volatility filter.
Long when Williams %R < -80 (oversold) with volume > 1.3x 20-period average and current ATR < 1.5x 20-period ATR average.
Short when Williams %R > -20 (overbought) with volume > 1.3x 20-period average and current ATR < 1.5x 20-period ATR average.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Uses discrete position sizing 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
Williams %R identifies extreme price levels; volume confirmation ensures participation; volatility filter avoids false signals in high volatility.
Works in bull markets (buying oversold dips) and bear markets (selling overbought rallies).
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
    
    # Get daily data for Williams %R, volume, and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
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
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ATR and Williams %R
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(atr_ma_20_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_1d_aligned[i]
        
        # Volatility filter: current ATR < 1.5x 20-period ATR average (breakout from low volatility)
        vol_filter = atr_aligned[i] < 1.5 * atr_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with volume and low volatility
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with volume and low volatility
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (moving out of oversold territory)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (moving out of overbought territory)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dWilliamsR_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0