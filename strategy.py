#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams Alligator with 12-hour trend filter.
Long when price > Alligator's Jaw, Teeth > Lips, and 12-hour EMA50 rising.
Short when price < Alligator's Jaw, Teeth < Lips, and 12-hour EMA50 falling.
Exit when price crosses Jaw or Teeth/Lips cross reverses.
Williams Alligator provides dynamic support/resistance; 12-hour EMA50 filters higher timeframe trend.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following 12h trend while using 4h Alligator for entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 12-hour data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Williams Alligator components (13, 8, 5 periods with future shifts)
    # Jaw (Blue Line): 13-period SMMA smoothed 8 bars ahead
    high_13 = pd.Series(high).rolling(window=13, min_periods=13).mean().values
    low_13 = pd.Series(low).rolling(window=13, min_periods=13).mean().values
    jaw_raw = (high_13 + low_13) / 2.0
    jaw = np.roll(jaw_raw, 8)  # shift forward 8 bars
    
    # Teeth (Red Line): 8-period SMMA smoothed 5 bars ahead
    high_8 = pd.Series(high).rolling(window=8, min_periods=8).mean().values
    low_8 = pd.Series(low).rolling(window=8, min_periods=8).mean().values
    teeth_raw = (high_8 + low_8) / 2.0
    teeth = np.roll(teeth_raw, 5)  # shift forward 5 bars
    
    # Lips (Green Line): 5-period SMMA smoothed 3 bars ahead
    high_5 = pd.Series(high).rolling(window=5, min_periods=5).mean().values
    low_5 = pd.Series(low).rolling(window=5, min_periods=5).mean().values
    lips_raw = (high_5 + low_5) / 2.0
    lips = np.roll(lips_raw, 3)  # shift forward 3 bars
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after enough data for Alligator (max shift is 8 bars)
    start_idx = max(13, 8) + 8  # 13 for jaw calculation + 8 for shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > Jaw, Teeth > Lips, and 12-hour EMA50 rising
            if (close[i] > jaw[i] and 
                teeth[i] > lips[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price < Jaw, Teeth < Lips, and 12-hour EMA50 falling
            elif (close[i] < jaw[i] and 
                  teeth[i] < lips[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below Jaw OR Teeth crosses below Lips
                if (close[i] < jaw[i] or 
                    teeth[i] < lips[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above Jaw OR Teeth crosses above Lips
                if (close[i] > jaw[i] or 
                    teeth[i] > lips[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Williams_Alligator_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0