#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Reversal_Trend_Volume
Hypothesis: 12h timeframe with Camarilla pivot reversals (R4/S4) combined with 1d trend filter and volume confirmation captures high-probability mean-reversion in range-bound markets and trend continuations in trending markets. Works in both bull and bear regimes by adapting to market structure. Targets 15-25 trades/year on 12h timeframe.
"""

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
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels for previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = high_1d[0]  # First day uses same day
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d[0]
    
    # Calculate pivot and Camarilla levels
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    R4 = close_prev + range_prev * 1.5000
    R3 = close_prev + range_prev * 1.2500
    R2 = close_prev + range_prev * 1.1666
    R1 = close_prev + range_prev * 1.0833
    S1 = close_prev - range_prev * 1.0833
    S2 = close_prev - range_prev * 1.1666
    S3 = close_prev - range_prev * 1.2500
    S4 = close_prev - range_prev * 1.5000
    
    # Align Camarilla levels to 12h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: >1.5x 30-period MA (approx 15 days on 12h)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(R4_aligned[i]) or
            np.isnan(S4_aligned[i]) or
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_30[i])
        
        # Entry logic: Reversal at extreme Camarilla levels with volume
        # Long when price touches S3/S4 in uptrend or S4 in downtrend (mean reversion)
        # Short when price touches R3/R4 in downtrend or R4 in uptrend (mean reversion)
        long_entry = vol_confirm and (
            (close[i] <= S3_aligned[i] and uptrend) or  # Mean reversion in uptrend
            (close[i] <= S4_aligned[i])                 # Strong support
        )
        
        short_entry = vol_confirm and (
            (close[i] >= R3_aligned[i] and downtrend) or  # Mean reversion in downtrend
            (close[i] >= R4_aligned[i])                   # Strong resistance
        )
        
        # Exit logic: Return to pivot or opposite extreme
        long_exit = close[i] >= pivot[i] or close[i] <= S4_aligned[i]
        short_exit = close[i] <= pivot[i] or close[i] >= R4_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_Pivot_Reversal_Trend_Volume"
timeframe = "12h"
leverage = 1.0