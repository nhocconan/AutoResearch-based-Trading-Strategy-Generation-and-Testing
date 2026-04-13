#!/usr/bin/env python3
"""
1D_1W_CAMARILLA_PIVOT_BREAKOUT_VOLUME_FILTER
Hypothesis: On the daily timeframe, Camarilla pivot levels derived from weekly candles provide strong support/resistance.
Breakouts above weekly R4 or below weekly S4 with volume confirmation (1.5x 20-day average volume) indicate institutional participation.
The weekly timeframe filters noise and captures major trend shifts, working in both bull and bear markets by trading in the direction of the breakout.
Target: 10-20 trades/year per symbol.
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
    
    # Get weekly data for Camarilla pivots
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each weekly bar
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Shift close by 1 to get previous week's close
    close_prev = np.roll(close_weekly, 1)
    close_prev[0] = close_weekly[0]  # first bar uses its own close
    
    range_weekly = high_weekly - low_weekly
    
    # Resistance levels
    R1 = close_prev + (range_weekly * 1.0833 / 12)
    R2 = close_prev + (range_weekly * 1.1666 / 6)
    R3 = close_prev + (range_weekly * 1.2500 / 4)
    R4 = close_prev + (range_weekly * 1.5000 / 2)
    
    # Support levels
    S1 = close_prev - (range_weekly * 1.0833 / 12)
    S2 = close_prev - (range_weekly * 1.1666 / 6)
    S3 = close_prev - (range_weekly * 1.2500 / 4)
    S4 = close_prev - (range_weekly * 1.5000 / 2)
    
    # Align levels to daily timeframe
    R4_aligned = align_htf_to_ltf(prices, df_weekly, R4)
    S4_aligned = align_htf_to_ltf(prices, df_weekly, S4)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above weekly R4 with volume expansion
        long_breakout = close[i] > R4_aligned[i] and volume_expansion[i]
        
        # Short breakdown: price breaks below weekly S4 with volume expansion
        short_breakout = close[i] < S4_aligned[i] and volume_expansion[i]
        
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

name = "1D_1W_Camarilla_Pivot_Breakout_Volume_Filter"
timeframe = "1d"
leverage = 1.0