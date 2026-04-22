#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams Alligator with 1-day EMA50 trend filter.
Long when price above Alligator teeth, jaws above teeth, lips above teeth, and 1-day EMA50 rising.
Short when price below Alligator teeth, jaws below teeth, lips below teeth, and 1-day EMA50 falling.
Exit when price crosses Alligator teeth or Alligator lines cross in opposite direction.
Williams Alligator identifies trend alignment; 1-day EMA50 filters higher timeframe trend.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following daily trend while using 6h Alligator for entries.
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
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator (13, 8, 5 periods with future shifts)
    # Jaw (Blue line): 13-period SMMA shifted 8 bars forward
    high_13 = pd.Series(high).rolling(window=13, min_periods=13).mean().values
    low_13 = pd.Series(low).rolling(window=13, min_periods=13).mean().values
    jaw_raw = (high_13 + low_13) / 2.0
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    
    # Teeth (Red line): 8-period SMMA shifted 5 bars forward
    high_8 = pd.Series(high).rolling(window=8, min_periods=8).mean().values
    low_8 = pd.Series(low).rolling(window=8, min_periods=8).mean().values
    teeth_raw = (high_8 + low_8) / 2.0
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    
    # Lips (Green line): 5-period SMMA shifted 3 bars forward
    high_5 = pd.Series(high).rolling(window=5, min_periods=5).mean().values
    low_5 = pd.Series(low).rolling(window=5, min_periods=5).mean().values
    lips_raw = (high_5 + low_5) / 2.0
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after enough data for Alligator
        # Skip if data not valid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above teeth, jaws above teeth, lips above teeth, and 1-day EMA50 rising
            if (close[i] > teeth[i] and 
                jaw[i] > teeth[i] and 
                lips[i] > teeth[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price below teeth, jaws below teeth, lips below teeth, and 1-day EMA50 falling
            elif (close[i] < teeth[i] and 
                  jaw[i] < teeth[i] and 
                  lips[i] < teeth[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below teeth OR jaws cross below teeth
                if (close[i] < teeth[i] or 
                    jaw[i] < teeth[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above teeth OR jaws cross above teeth
                if (close[i] > teeth[i] or 
                    jaw[i] > teeth[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Williams_Alligator_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0