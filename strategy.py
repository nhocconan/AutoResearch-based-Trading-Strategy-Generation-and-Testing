#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike for trend strength and momentum.
# Uses Williams Alligator (Jaw, Teeth, Lips) for trend direction, Elder Ray (Bull/Bear Power) for momentum,
# and volume spike for confirmation. Enters only when all three align.
# Targets 20-50 trades/year (80-200 total over 4 years) with strict entry conditions.
# Works in bull/bear by following trend strength and momentum.
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
    
    # Williams Alligator: SMoothed Moving Average (SMMA) with specific periods
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(source, length):
        sma = pd.Series(source).rolling(window=length, min_periods=length).mean()
        # SMMA is essentially EMA with alpha = 1/length
        return sma.ewm(alpha=1/length, adjust=False).mean().values
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Alligator aligned (Lips > Teeth > Jaw) AND Bull Power > 0 AND Volume Spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                bull_power[i] > 0 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned (Lips < Teeth < Jaw) AND Bear Power < 0 AND Volume Spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  bear_power[i] < 0 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator reverses OR Bear Power becomes positive
            if (lips[i] < teeth[i] or bear_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator reverses OR Bull Power becomes negative
            if (lips[i] > teeth[i] or bull_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals