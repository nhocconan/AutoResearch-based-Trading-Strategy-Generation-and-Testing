#!/usr/bin/env python3
# 6h_1d_williams_alligator_v1
# Strategy: 6s Williams Alligator with 1d trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Williams Alligator (3 SMAs: Jaw, Teeth, Lips) identifies trend direction and strength.
# When the Alligator is "awake" (lines separated and aligned), trade in direction of alignment.
# Combined with 1d trend filter (price above/below EMA50) to ensure higher timeframe alignment.
# Designed for low frequency (15-35 trades/year) to minimize fee drag in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_williams_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: 3 SMAs with different periods and shifts
    # Jaw: 13-period SMA, shifted 8 bars forward
    # Teeth: 8-period SMA, shifted 5 bars forward  
    # Lips: 5-period SMA, shifted 3 bars forward
    close_series = pd.Series(close)
    
    jaw_raw = close_series.rolling(window=13, min_periods=13).mean()
    teeth_raw = close_series.rolling(window=8, min_periods=8).mean()
    lips_raw = close_series.rolling(window=5, min_periods=5).mean()
    
    # Apply shifts (Alligator definition: future-shifted SMAs)
    jaw = jaw_raw.shift(8)
    teeth = teeth_raw.shift(5)
    lips = lips_raw.shift(3)
    
    # Alligator values aligned to current bar (no look-ahead)
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_values[i]) or 
            np.isnan(teeth_values[i]) or np.isnan(lips_values[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Alligator conditions: check alignment and separation
        # Bullish alignment: Lips > Teeth > Jaw (all separated)
        bullish_aligned = (lips_values[i] > teeth_values[i]) and (teeth_values[i] > jaw_values[i])
        # Bearish alignment: Lips < Teeth < Jaw (all separated)
        bearish_aligned = (lips_values[i] < teeth_values[i]) and (teeth_values[i] < jaw_values[i])
        
        # Entry logic: Alligator alignment + trend filter
        if bullish_aligned and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_aligned and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Alligator sleeping (lines intertwined) or trend change
        elif position == 1 and (not bullish_aligned or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish_aligned or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals