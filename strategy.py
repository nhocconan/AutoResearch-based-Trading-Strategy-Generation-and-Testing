#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Alligator lines: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA).
# Enter long when Lips > Teeth > Jaw and all rising, with 1d EMA(50) uptrend and volume expansion.
# Enter short when Lips < Teeth < Jaw and all falling, with 1d EMA(50) downtrend and volume expansion.
# Uses SMMA (Smoothed Moving Average) for Alligator lines.
# Designed for 15-30 trades/year on 4h timeframe with focus on trend strength.
# Volume filter ensures breakouts have conviction, reducing false signals.
# 1d trend filter prevents counter-trend trading in choppy markets.

name = "4h_1d_alligator_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    if length < 1:
        return source
    smma = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple average
    smma[length-1] = np.mean(source[:length])
    # Subsequent values: (prev_smma * (length-1) + current) / length
    for i in range(length, len(source)):
        smma[i] = (smma[i-1] * (length-1) + source[i]) / length
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Alligator lines using SMMA
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    lips = smma(close, 5)   # Lips: 5-period SMMA
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after Jaw period
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * 20-period average volume
        vol_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Determine 1d trend direction
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Alligator conditions: alignment and slope
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Slope conditions (1-bar change)
        lips_rising = lips[i] > lips[i-1]
        teeth_rising = teeth[i] > teeth[i-1]
        jaw_rising = jaw[i] > jaw[i-1]
        lips_falling = lips[i] < lips[i-1]
        teeth_falling = teeth[i] < teeth[i-1]
        jaw_falling = jaw[i] < jaw[i-1]
        
        # Entry conditions
        bullish_entry = (lips_above_teeth and teeth_above_jaw and 
                        lips_rising and teeth_rising and jaw_rising and 
                        vol_filter and is_uptrend)
        bearish_entry = (lips_below_teeth and teeth_below_jaw and 
                        lips_falling and teeth_falling and jaw_falling and 
                        vol_filter and is_downtrend)
        
        # Exit conditions: loss of alignment
        exit_long = not (lips_above_teeth and teeth_above_jaw)
        exit_short = not (lips_below_teeth and teeth_below_jaw)
        
        # Priority: entry > exit > hold
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals