#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike
# Uses Williams Alligator (SMAs with offset) for trend direction,
# Elder Ray (bull/bear power) for momentum confirmation,
# Volume spike for entry confirmation.
# Works in bull/bear by following Alligator trend and filtering with Elder Ray.
# Targets 20-50 trades/year (80-200 total over 4 years) with strict entry conditions.
name = "4h_WilliamsAlligator_ElderRay_Volume"
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
    
    # Williams Alligator: Jaw (13-period SMA offset 8), Teeth (8-period SMA offset 5), Lips (5-period SMA offset 3)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_filter = volume > (volume_ma * 2.0)
    
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
            # Long: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND Volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                bull_power[i] > 0 and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (bearish alignment) AND Bear Power < 0 AND Volume spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  bear_power[i] < 0 and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator turns bearish (Lips < Teeth OR Teeth < Jaw) OR Bear Power < 0
            if lips[i] < teeth[i] or teeth[i] < jaw[i] or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator turns bullish (Lips > Teeth OR Teeth > Jaw) OR Bull Power > 0
            if lips[i] > teeth[i] or teeth[i] > jaw[i] or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals