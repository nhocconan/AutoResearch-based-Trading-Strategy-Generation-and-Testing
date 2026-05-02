#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Williams Alligator (JAW/TEETH/LIPS) identifies trend absence (alligator sleeping) vs presence (awakening with mouth open)
# In ranging markets (alligator sleeping): Jaw, Teeth, Lips intertwined → no trades
# In trending markets (alligator awakening): Lips > Teeth > Jaw (bull) or Lips < Teeth < Jaw (bear) → trend follow
# 1w EMA50 provides higher timeframe trend filter to align with dominant momentum and reduce counter-trend whipsaws
# Volume spike (2.0x 20-period average) confirms breakout conviction
# Targets 50-150 trades over 4 years (12-37/year) for 12h timeframe
# Works in both bull and bear markets by only trading when alligator is awake (trending) and aligned with 1w trend

name = "12h_Williams_Alligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on 12h timeframe (using current timeframe data)
    # Jaw: 13-period SMMA shifted 8 bars ahead
    # Teeth: 8-period SMMA shifted 5 bars ahead  
    # Lips: 5-period SMMA shifted 3 bars ahead
    # SMMA = Smoothed Moving Average (similar to EMA but with different smoothing)
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    # Calculate SMMA using EMA as approximation (common implementation)
    jaw = pd.Series(close).ewm(span=jaw_period, adjust=False).mean().values
    teeth = pd.Series(close).ewm(span=teeth_period, adjust=False).mean().values
    lips = pd.Series(close).ewm(span=lips_period, adjust=False).mean().values
    
    # Apply shifts (using NaN for unavailable values)
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) > jaw_shift:
        jaw_shifted[jaw_shift:] = jaw[:-jaw_shift]
    if len(teeth) > teeth_shift:
        teeth_shifted[teeth_shift:] = teeth[:-teeth_shift]
    if len(lips) > lips_shift:
        lips_shifted[lips_shift:] = lips[:-lips_shift]
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all indicators)
    start_idx = max(jaw_shift, teeth_shift, lips_shift, 20) + 5
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator awake: Lips > Teeth > Jaw (bullish alignment) OR Lips < Teeth < Jaw (bearish alignment)
            bullish_alignment = lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]
            bearish_alignment = lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]
            
            # Long: Bullish alignment + price > 1w EMA50 + volume spike
            if bullish_alignment and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price < 1w EMA50 + volume spike
            elif bearish_alignment and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator starts sleeping (Lips crosses below Teeth) OR trend reversal
            if lips_shifted[i] < teeth_shifted[i]:  # Alligator closing mouth - losing momentum
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator starts sleeping (Lips crosses above Teeth) OR trend reversal
            if lips_shifted[i] > teeth_shifted[i]:  # Alligator closing mouth - losing momentum
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals