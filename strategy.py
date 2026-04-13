#!/usr/bin/env python3
"""
4h_1d_Camarilla_3T4_Reversal
Hypothesis: Price reverses from Camarilla H3/L3 levels with trend filter (HMA21) and volume confirmation.
Works in bull markets via bounces from L3 and bear markets via rejections at H3. Target: 25-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average."""
    half = int(period / 2)
    sqrt = int(np.sqrt(period))
    wma2 = pd.Series(series).ewm(span=half, adjust=False).mean()
    wma1 = pd.Series(series).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma2 - wma1
    hma = pd.Series(raw_hma).ewm(span=sqrt, adjust=False).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d trend filter: HMA21
    hma_21 = calculate_hma(close_1d, 21)
    uptrend = hma_21 > 0  # HMA slope positive
    downtrend = hma_21 < 0  # HMA slope negative
    
    # Calculate Camarilla H3/L3 on daily
    range_1d = high_1d - low_1d
    range_1d = np.where(range_1d == 0, 1e-10, range_1d)
    C = close_1d
    H3 = C + ((high_1d - low_1d) * 1.1666)
    L3 = C - ((high_1d - low_1d) * 1.1666)
    
    # Align all data to 4h timeframe
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or 
            np.isnan(uptrend_aligned[i]) or np.isnan(downtrend_aligned[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: bounce from L3 in uptrend with volume expansion
        long_condition = (low[i] <= L3_1d_aligned[i]) and uptrend_aligned[i] > 0.5 and volume_expansion[i]
        
        # Short: rejection at H3 in downtrend with volume expansion
        short_condition = (high[i] >= H3_1d_aligned[i]) and downtrend_aligned[i] > 0.5 and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif long_condition and position == 1:
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_condition and position == -1:
            signals[i] = -position_size
        else:
            # Exit: reverse signal or loss of trend/volume
            if position == 1 and (not uptrend_aligned[i] > 0.5 or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (not downtrend_aligned[i] > 0.5 or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Camarilla_3T4_Reversal"
timeframe = "4h"
leverage = 1.0