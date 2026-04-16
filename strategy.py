#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with 1-week EMA50 trend filter
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA50
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA50
# Exit when alignment breaks or price crosses 8-period EMA (signal line)
# Williams Alligator identifies trend phases; 1w EMA50 filters counter-trend trades
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag on 6h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1-week EMA50 (trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 6h Williams Alligator (13,8,5 SMAs shifted) ===
    df_6h = get_htf_data(prices, '6h')
    # Jaw (13-period SMA, shifted 8 bars)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    # Teeth (8-period SMA, shifted 5 bars)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    # Lips (5-period SMA, shifted 3 bars)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # === 6h EMA8 (exit signal) ===
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_8[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_8_val = ema_8[i]
        ema_50_val = ema_50_1w_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit if Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw) OR price crosses below EMA8
            if not (lips_val > teeth_val and teeth_val > jaw_val) or price < ema_8_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit if Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw) OR price crosses above EMA8
            if not (lips_val < teeth_val and teeth_val < jaw_val) or price > ema_8_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw AND price above 1w EMA50
            if lips_val > teeth_val and teeth_val > jaw_val and price > ema_50_val:
                signals[i] = 0.25
                position = 1
                continue
            # Bearish alignment: Lips < Teeth < Jaw AND price below 1w EMA50
            elif lips_val < teeth_val and teeth_val < jaw_val and price < ema_50_val:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_1wEMA50_TrendFilter"
timeframe = "6h"
leverage = 1.0