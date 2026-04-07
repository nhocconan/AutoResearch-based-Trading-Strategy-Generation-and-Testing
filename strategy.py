#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator with daily trend filter and volume confirmation
# Alligator uses smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
# Daily trend filter ensures trades align with higher timeframe direction.
# Volume confirmation filters low-participation moves. Designed for low frequency in 4h timeframe.
# Works in bull markets (buy when Lips cross above Teeth/Jaw in uptrend) and bear markets (sell when Lips cross below in downtrend).

name = "4h_alligator_daily_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate daily EMA13 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema13_1d = close_1d.ewm(span=13, min_periods=13, adjust=False).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Alligator components (SMMA = smoothed moving average)
    close_s = pd.Series(close)
    # Jaw: 13-period SMMA, 8 bars ahead
    jaw = close_s.rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA, 5 bars ahead
    teeth = close_s.rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA, 3 bars ahead
    lips = close_s.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after Alligator warmup
        # Skip if required data not available
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Daily trend: close above/below daily EMA13
        daily_uptrend = close[i] > ema13_1d_aligned[i]
        daily_downtrend = close[i] < ema13_1d_aligned[i]
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Alligator signals
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if daily trend turns down or Alligator turns bearish
            if not daily_uptrend or not (lips_above_teeth and teeth_above_jaw):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if daily trend turns up or Alligator turns bullish
            if not daily_downtrend or not (lips_below_teeth and teeth_below_jaw):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: daily uptrend + Alligator bullish (Lips > Teeth > Jaw) + volume confirmation
            if daily_uptrend and lips_above_teeth and teeth_above_jaw and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: daily downtrend + Alligator bearish (Lips < Teeth < Jaw) + volume confirmation
            elif daily_downtrend and lips_below_teeth and teeth_below_jaw and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals