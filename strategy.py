#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and entry signals.
# Trend filter: 1-week EMA50 - only trade in direction of weekly trend.
# Volume confirmation: current volume > 1.5x median of last 20 periods.
# Designed for fewer trades (target 50-150/year) to avoid fee drag on 12h timeframe.
# Works in bull markets (buy when Lips cross above Teeth/Jaw in uptrend) 
# and bear markets (sell when Lips cross below Teeth/Jaw in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator (13,8,5 SMAs with future shifts)
    # Jaw: 13-period SMMA shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    # Teeth: 8-period SMMA shifted 5 bars forward  
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    # Lips: 5-period SMMA shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # Calculate 1-week EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size - 25% of capital
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            continue
            
        # Determine weekly trend direction
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Long entry: Lips cross above Teeth AND Jaw (bullish alignment) in weekly uptrend
        # Plus volume confirmation
        if (lips_aligned[i] > teeth_aligned[i] and lips_aligned[i] > jaw_aligned[i] and
            lips_aligned[i-1] <= teeth_aligned[i-1] and  # Crossed above this bar
            weekly_uptrend and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
            
        # Short entry: Lips cross below Teeth AND Jaw (bearish alignment) in weekly downtrend
        # Plus volume confirmation
        elif (lips_aligned[i] < teeth_aligned[i] and lips_aligned[i] < jaw_aligned[i] and
              lips_aligned[i-1] >= teeth_aligned[i-1] and  # Crossed below this bar
              weekly_downtrend and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
            
        # Exit: Lips cross back to opposite side of Teeth or weekly trend changes
        elif position == 1 and (lips_aligned[i] < teeth_aligned[i] or not weekly_uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (lips_aligned[i] > teeth_aligned[i] or not weekly_downtrend):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0