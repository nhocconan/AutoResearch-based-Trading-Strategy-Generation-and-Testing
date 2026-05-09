#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 12h EMA trend filter + volume confirmation
# Williams Alligator (Jaw, Teeth, Lips) identifies trends when lines are aligned and separated.
# In uptrend: Lips > Teeth > Jaw; in downtrend: Lips < Teeth < Jaw.
# 12h EMA ensures higher timeframe trend alignment.
# Volume confirms participation. Works in both bull (Alligator up) and bear (Alligator down) markets.
name = "6h_WilliamsAlligator_12hEMA_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator parameters (13,8,5)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Jaw (blue line): 13-period SMMA shifted 8 bars
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean()
    jaw = jaw.shift(jaw_shift)
    
    # Teeth (red line): 8-period SMMA shifted 5 bars
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean()
    teeth = teeth.shift(teeth_shift)
    
    # Lips (green line): 5-period SMMA shifted 3 bars
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean()
    lips = lips.shift(lips_shift)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period, teeth_period, lips_period) + max(jaw_shift, teeth_shift, lips_shift)
    start_idx = max(start_idx, 50)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(jaw_vals[i]) or np.isnan(teeth_vals[i]) or np.isnan(lips_vals[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (Alligator up) + price > 12h EMA50 + volume confirmation
            if (lips_vals[i] > teeth_vals[i] and teeth_vals[i] > jaw_vals[i] and 
                price > ema_12h_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (Alligator down) + price < 12h EMA50 + volume confirmation
            elif (lips_vals[i] < teeth_vals[i] and teeth_vals[i] < jaw_vals[i] and 
                  price < ema_12h_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator convergence (Lips < Teeth) or price < 12h EMA50
            if lips_vals[i] < teeth_vals[i] or price < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator convergence (Lips > Teeth) or price > 12h EMA50
            if lips_vals[i] > teeth_vals[i] or price > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals