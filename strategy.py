#!/usr/bin/env python3
"""
4h_12h_Combined_Trend_Breakout
Hypothesis: 4h Donchian breakout with 12h trend filter (HMA21) and volume confirmation.
Works in bull markets via breakouts above upper band and in bear markets via breakdowns below lower band.
Volume ensures institutional participation. Target: 25-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Calculate Hull Moving Average."""
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
    wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
    raw = 2 * wma2 - wma1
    hma = pd.Series(raw).ewm(span=sqrt, adjust=False).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channels (20-period)
    high_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for trend filter (HMA21)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    hma_21 = calculate_hma(close_12h, 21)
    hma_21_prev = np.roll(hma_21, 1)
    hma_21_prev[0] = hma_21[0]
    uptrend = hma_21 > hma_21_prev
    downtrend = hma_21 < hma_21_prev
    
    # Align all data to 4h timeframe
    high_max_aligned = align_htf_to_ltf(prices, df_4h, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_4h, low_min)
    uptrend_aligned = align_htf_to_ltf(prices, df_12h, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_12h, downtrend.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(high_max_aligned[i]) or np.isnan(low_min_aligned[i]) or 
            np.isnan(uptrend_aligned[i]) or np.isnan(downtrend_aligned[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above upper Donchian in uptrend with volume expansion
        long_condition = (close[i] > high_max_aligned[i]) and uptrend_aligned[i] > 0.5 and volume_expansion[i]
        
        # Short: breakdown below lower Donchian in downtrend with volume expansion
        short_condition = (close[i] < low_min_aligned[i]) and downtrend_aligned[i] > 0.5 and volume_expansion[i]
        
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
            # Exit conditions: reverse signal or loss of trend/volume
            if position == 1 and (not uptrend_aligned[i] > 0.5 or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (not downtrend_aligned[i] > 0.5 or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_12h_Combined_Trend_Breakout"
timeframe = "4h"
leverage = 1.0