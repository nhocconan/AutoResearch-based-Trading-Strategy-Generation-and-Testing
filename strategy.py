#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination
# ADX > 25 identifies strong trends while Alligator (Jaw/Teeth/Lips) confirms direction.
# Long when ADX > 25, Lips > Teeth > Jaw (bullish alignment)
# Short when ADX > 25, Lips < Teeth < Jaw (bearish alignment)
# Exit when ADX < 20 (trend weakening) or Alligator lines cross (signal reversal)
# Uses smoothed SMAs to reduce whipsaw. Targets 20-40 trades/year for low frequency.
name = "6h_ADX_Alligator_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator (SMAs with specific periods)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Shift jaws/teeth/lips forward as per Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # ADX calculation
    period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First TR has no previous close
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period, teeth_period, lips_period, period) + 8  # Account for shifts
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            if (adx[i] > 25 and 
                lips[i] > teeth[i] and 
                teeth[i] > jaw[i]):
                signals[i] = 0.25
                position = 1
            # Bearish alignment: Lips < Teeth < Jaw
            elif (adx[i] > 25 and 
                  lips[i] < teeth[i] and 
                  teeth[i] < jaw[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Exit: ADX weakening (<20) or Alligator cross (Lips < Teeth)
            if (adx[i] < 20) or (lips[i] < teeth[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Exit: ADX weakening (<20) or Alligator cross (Lips > Teeth)
            if (adx[i] < 20) or (lips[i] > teeth[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals