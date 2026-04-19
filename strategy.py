#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike for trend strength
# Uses Williams Alligator (Jaw/Teeth/Lips) for trend direction, Elder Ray (Bull/Bear Power) for momentum,
# and volume confirmation to filter false signals. Designed to work in both bull and bear markets
# by capturing strong trends with clear entry/exit rules. Targets 20-40 trades/year (80-160 total over 4 years).
name = "4h_WilliamsAlligator_ElderRay_Volume_Spike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMoothed Moving Averages (using SMA as proxy for SMMA)
    jaw_period, teeth_period, lips_period = 13, 8, 5
    jaw_shift, teeth_shift, lips_shift = 8, 5, 3
    
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(jaw_shift).values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(teeth_shift).values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().shift(lips_shift).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: volume > 2.0 * 20-period average (spike detection)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period + jaw_shift, teeth_period + teeth_shift, lips_period + lips_shift, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND Volume Spike
            if (lips[i] > teeth[i] > jaw[i] and 
                bull_power[i] > 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND Bear Power < 0 AND Volume Spike
            elif (lips[i] < teeth[i] < jaw[i] and 
                  bear_power[i] < 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Lips < Teeth (loss of bullish alignment) OR Bear Power > 0 (bullish momentum fails)
            if lips[i] < teeth[i] or bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Lips > Teeth (loss of bearish alignment) OR Bull Power < 0 (bearish momentum fails)
            if lips[i] > teeth[i] or bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals