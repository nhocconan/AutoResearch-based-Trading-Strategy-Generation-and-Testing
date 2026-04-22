#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams Alligator with 1-day EMA50 trend and volume spike.
Long when price above Alligator teeth (green line) with upward alignment and 1-day EMA50 rising.
Short when price below Alligator teeth with downward alignment and 1-day EMA50 falling.
Exit when price crosses Alligator jaws (red line).
Alligator uses SMAs (13,8,5) with future shift (8,5,3) for proper alignment.
Williams Alligator identifies trend phases; 1-day EMA50 filters trend direction;
volume spike confirms institutional participation. Designed for low trade frequency by requiring
multiple confirmations. Works in both bull and bear markets by following the daily trend.
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
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator (SMAs with future shift)
    # Jaw (blue): SMA(13) shifted 8 bars forward
    # Teeth (red): SMA(8) shifted 5 bars forward  
    # Lips (green): SMA(5) shifted 3 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    
    # Apply forward shift to avoid look-ahead (Alligator's predictive nature)
    jaw_shifted = jaw.shift(8)   # SMA(13) shifted 8 bars
    teeth_shifted = teeth.shift(5) # SMA(8) shifted 5 bars
    lips_shifted = lips.shift(3)   # SMA(5) shifted 3 bars
    
    # Align 1-day EMA50 to 4h timeframe
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(jaw_shifted.iloc[i]) or np.isnan(teeth_shifted.iloc[i]) or np.isnan(lips_shifted.iloc[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Alligator alignment checks
        teeth_val = teeth_shifted.iloc[i]
        lips_val = lips_shifted.iloc[i]
        jaw_val = jaw_shifted.iloc[i]
        
        # Upward alignment: Lips > Teeth > Jaw (green > red > blue)
        upward_aligned = lips_val > teeth_val > jaw_val
        # Downward alignment: Lips < Teeth < Jaw (green < red < blue)
        downward_aligned = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Long: Price above teeth with upward alignment and 1-day EMA50 rising
            if (close[i] > teeth_val and upward_aligned and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price below teeth with downward alignment and 1-day EMA50 falling
            elif (close[i] < teeth_val and downward_aligned and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses jaws (jaw line)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below jaw
                if close[i] < jaw_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above jaw
                if close[i] > jaw_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Williams_Alligator_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0