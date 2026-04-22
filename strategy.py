#!/usr/bin/env python3

"""
Hypothesis: 4-hour price action strategy using 1-day pivot points (support/resistance) 
combined with volume confirmation and trend filter. Enters long at support in uptrend,
short at resistance in downtrend only when volume exceeds 2x the 20-period average.
Uses fixed position sizing (0.25) to limit risk. Designed for 15-30 trades/year 
to avoid fee drag while capturing meaningful reversals in both bull and bear markets.
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
    
    # Load 1d data for pivot points and trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day pivot points (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_point = (high_1d + low_1d + close_1d) / 3.0
    resistance_1 = 2 * pivot_point - low_1d
    support_1 = 2 * pivot_point - high_1d
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    resistance_1_aligned = align_htf_to_ltf(prices, df_1d, resistance_1)
    support_1_aligned = align_htf_to_ltf(prices, df_1d, support_1)
    
    # 1-day EMA for trend filter (34-period)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(resistance_1_aligned[i]) or 
            np.isnan(support_1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price at or below support in uptrend (above EMA)
            if close[i] <= support_1_aligned[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price at or above resistance in downtrend (below EMA)
            elif close[i] >= resistance_1_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches pivot or trend reverses
                if close[i] >= pivot_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reaches pivot or trend reverses
                if close[i] <= pivot_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Pivot_SupportResistance_Volume"
timeframe = "4h"
leverage = 1.0