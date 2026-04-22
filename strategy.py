#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator uses three smoothed moving averages (Jaw: 13-period SMA shifted 8 bars forward,
# Teeth: 8-period SMA shifted 5 bars forward, Lips: 5-period SMA shifted 3 bars forward).
# In trending markets, these lines are well-separated and ordered.
# In ranging markets, they intertwine.
# We use 1d EMA50 for trend direction (bullish if price > EMA50, bearish if price < EMA50).
# Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume spike.
# Entry: Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume spike.
# Exit: When Alligator lines re-intertwine (Lips crosses Teeth or Jaw) OR volume drops.
# Designed to catch strong trends while avoiding chop. Targets 15-30 trades/year on 12h.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for EMA50 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams Alligator components on 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Jaw: 13-period SMMA shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    
    # Teeth: 8-period SMMA shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Lips: 5-period SMMA shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    # Convert to numpy arrays, filling NaN with 0 for alignment
    jaw = jaw.fillna(0).values
    teeth = teeth.fillna(0).values
    lips = lips.fillna(0).values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema50 = ema50_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # Alligator alignment conditions
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Look for new entries
            if bullish_alignment and price > ema50 and vol_spike:
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and price < ema50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Check for exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Alligator lines re-intertwine (Lips crosses below Teeth or Jaw)
                if lips_val <= teeth_val or lips_val <= jaw_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Alligator lines re-intertwine (Lips crosses above Teeth or Jaw)
                if lips_val >= teeth_val or lips_val >= jaw_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0