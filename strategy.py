#!/usr/bin/env python3
"""
Hypothesis: 6-hour Chandelier Exit with 1-day ATR trend filter.
Long when price > Chandelier Exit (long) and 1-day ATR > 20-period average ATR.
Short when price < Chandelier Exit (short) and 1-day ATR > 20-period average ATR.
Exit when price crosses Chandelier Exit or 1-day ATR drops below average.
Chandelier Exit adapts to volatility, ATR filter ensures trending conditions.
Works in both bull and bear markets by following volatility-adjusted trends while filtering for strong moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1-day data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for 1-day
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    atr_1d_avg = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_1d_avg_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_avg)
    
    # Chandelier Exit for 6h (22-period, 3.0 multiplier)
    atr_period = 22
    mult = 3.0
    
    # Calculate ATR for 6h
    tr1_6h = np.abs(high - low)
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr2_6h[0] = tr1_6h[0]
    tr3_6h[0] = tr1_6h[0]
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate Chandelier Exit
    chand_long = np.full(n, np.nan)
    chand_short = np.full(n, np.nan)
    
    for i in range(atr_period, n):
        # Long exit: highest high - ATR * mult
        highest_high = np.max(high[i-atr_period+1:i+1])
        chand_long[i] = highest_high - (atr_6h[i] * mult)
        
        # Short exit: lowest low + ATR * mult
        lowest_low = np.min(low[i-atr_period+1:i+1])
        chand_short[i] = lowest_low + (atr_6h[i] * mult)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(chand_long[i]) or np.isnan(chand_short[i]) or np.isnan(atr_1d_avg_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above Chandelier Long and 1-day ATR above average
            if close[i] > chand_long[i] and atr_1d[i] > atr_1d_avg_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below Chandelier Short and 1-day ATR above average
            elif close[i] < chand_short[i] and atr_1d[i] > atr_1d_avg_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below Chandelier Long
                if close[i] < chand_long[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above Chandelier Short
                if close[i] > chand_short[i]:
                    exit_signal = True
            
            # Also exit if 1-day ATR drops below average (weakening trend)
            if atr_1d[i] <= atr_1d_avg_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Chandelier_Exit_1dATR_Trend_Filter"
timeframe = "6h"
leverage = 1.0