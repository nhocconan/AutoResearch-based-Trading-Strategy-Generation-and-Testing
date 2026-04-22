#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams Alligator with 1-day trend filter.
Long when price is above the Alligator's lips, jaw > teeth > lips (bullish alignment), and 1-day EMA50 rising.
Short when price is below the Alligator's lips, jaw < teeth < lips (bearish alignment), and 1-day EMA50 falling.
Exit when price crosses the Alligator's teeth or the alignment reverses.
The Alligator identifies trend phases; 1-day EMA50 filters higher timeframe trend.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following daily trend while using 4h Alligator for entries.
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
    
    # Williams Alligator parameters (13, 8, 5 periods, smoothed with future shift)
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    # SMMA (Smoothed Moving Average) is similar to EMA but with different smoothing
    
    # Calculate SMMA using EMA as approximation (common practice)
    # For simplicity, we'll use EMA with the same period
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Apply forward shift to avoid look-ahead (Alligator's future shift)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # First values will be invalid due to roll, handled by nan checks
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after enough data for Alligator
        # Skip if data not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: jaw > teeth > lips
            bullish_alignment = jaw_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > lips_shifted[i]
            # Bearish alignment: jaw < teeth < lips
            bearish_alignment = jaw_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < lips_shifted[i]
            
            # Long: Price above lips, bullish alignment, and 1-day EMA50 rising
            if (close[i] > lips_shifted[i] and 
                bullish_alignment and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price below lips, bearish alignment, and 1-day EMA50 falling
            elif (close[i] < lips_shifted[i] and 
                  bearish_alignment and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below lips OR alignment turns bearish
                if (close[i] < lips_shifted[i] or 
                    jaw_shifted[i] < teeth_shifted[i] or 
                    teeth_shifted[i] < lips_shifted[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above lips OR alignment turns bullish
                if (close[i] > lips_shifted[i] or 
                    jaw_shifted[i] > teeth_shifted[i] or 
                    teeth_shifted[i] > lips_shifted[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Williams_Alligator_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0