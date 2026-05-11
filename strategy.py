#!/usr/bin/env python3
"""
12h_1w_Alligator_Turn_Backtest
Hypothesis: Williams Alligator on weekly timeframe defines trend direction (jaw-teeth-lips alignment).
On 12h chart, enter long when price crosses above teeth in bullish alignment (jaw < teeth < lips),
enter short when price crosses below teeth in bearish alignment (jaw > teeth > lips).
Exit when price crosses back across teeth or Alligator alignment breaks.
Uses Williams Alligator (SMMA with specific periods) on weekly data for trend filter.
Targets 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
Works in both bull/bear by following Alligator's trend definition.
"""

name = "12h_1w_Alligator_Turn"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's smoothing"""
    if len(source) == 0:
        return np.array([])
    result = np.full_like(source, np.nan, dtype=np.float64)
    alpha = 1.0 / length
    result[0] = source[0]
    for i in range(1, len(source)):
        if np.isnan(source[i]):
            result[i] = result[i-1]
        else:
            result[i] = (1 - alpha) * result[i-1] + alpha * source[i]
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Alligator calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:  # Need enough for SMMA(13,8,5)
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # --- Weekly Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) ---
    # All lines are SMMA of median price (hl2)
    median_price = (df_1w['high'].values + df_1w['low'].values) / 2.0
    
    # Jaw: Blue line - SMMA(median, 13) offset 8 bars
    jaw_raw = smma(median_price, 13)
    jaw = np.roll(jaw_raw, 8)  # Shift forward 8 bars
    
    # Teeth: Red line - SMMA(median, 8) offset 5 bars  
    teeth_raw = smma(median_price, 8)
    teeth = np.roll(teeth_raw, 5)  # Shift forward 5 bars
    
    # Lips: Green line - SMMA(median, 5) offset 3 bars
    lips_raw = smma(median_price, 5)
    lips = np.roll(lips_raw, 3)  # Shift forward 3 bars
    
    # Align Alligator lines to 12h timeframe (wait for weekly bar close)
    jaw_12h = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_12h = align_htf_to_ltf(prices, df_1w, teeth)
    lips_12h = align_htf_to_ltf(prices, df_1w, lips)
    
    # Determine Alligator alignment (wait for weekly close)
    # Bullish: jaw < teeth < lips (all lines ascending, price above)
    # Bearish: jaw > teeth > lips (all lines descending, price below)
    bullish_alignment = (jaw_12h < teeth_12h) & (teeth_12h < lips_12h)
    bearish_alignment = (jaw_12h > teeth_12h) & (teeth_12h > lips_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period for SMMA calculations
    start_idx = 35  # Enough for all SMMA calculations
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or 
            np.isnan(lips_12h[i]) or np.isnan(bullish_alignment[i]) or
            np.isnan(bearish_alignment[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for entries only in direction of Alligator alignment
            # Long: price crosses above teeth in bullish alignment
            if (close_12h[i] > teeth_12h[i] and 
                close_12h[i-1] <= teeth_12h[i-1] and 
                bullish_alignment[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below teeth in bearish alignment
            elif (close_12h[i] < teeth_12h[i] and 
                  close_12h[i-1] >= teeth_12h[i-1] and 
                  bearish_alignment[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses back below teeth OR alignment breaks
                if (close_12h[i] < teeth_12h[i] and 
                    close_12h[i-1] >= teeth_12h[i-1]) or not bullish_alignment[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses back above teeth OR alignment breaks
                if (close_12h[i] > teeth_12h[i] and 
                    close_12h[i-1] <= teeth_12h[i-1]) or not bearish_alignment[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals