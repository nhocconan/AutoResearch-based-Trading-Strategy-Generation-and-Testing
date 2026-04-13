#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: Camarilla pivot levels from daily candles provide robust support/resistance. 
Breakouts above R4 or below S4 with volume expansion and alignment to 12h trend (via EMA50) capture institutional moves. 
Works in bull/bear by following trend direction. Target: 12-30 trades/year.
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
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous close for pivot calculation
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]
    
    range_1d = high_1d - low_1d
    
    # Resistance levels
    R1 = close_prev + (range_1d * 1.0833 / 12)
    R2 = close_prev + (range_1d * 1.1666 / 6)
    R3 = close_prev + (range_1d * 1.2500 / 4)
    R4 = close_prev + (range_1d * 1.5000 / 2)
    
    # Support levels
    S1 = close_prev - (range_1d * 1.0833 / 12)
    S2 = close_prev - (range_1d * 1.1666 / 6)
    S3 = close_prev - (range_1d * 1.2500 / 4)
    S4 = close_prev - (range_1d * 1.5000 / 2)
    
    # Align Camarilla levels to 12h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price above R4, volume expansion, and above daily EMA50 (uptrend)
        long_breakout = (close[i] > R4_aligned[i]) and volume_expansion[i] and (close[i] > ema50_1d_aligned[i])
        
        # Short breakdown: price below S4, volume expansion, and below daily EMA50 (downtrend)
        short_breakout = (close[i] < S4_aligned[i]) and volume_expansion[i] and (close[i] < ema50_1d_aligned[i])
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0