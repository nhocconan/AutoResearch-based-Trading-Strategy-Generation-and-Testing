#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with Elder Ray and volume confirmation.
# Uses Williams Alligator (Jaws, Teeth, Lips) for trend direction and Elder Ray (Bull/Bear Power) for momentum.
# Enters only during 08-20 UTC session to avoid low-volume noise.
# Targets 20-30 trades/year (80-120 total over 4 years) with strict entry conditions.
# Works in bull/bear by following Williams Alligator trend alignment.
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
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Williams Alligator: SMAs with specific periods and offsets
    # Jaws: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars  
    # Lips: 5-period SMA, shifted 3 bars
    jaws = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Alligator aligned (Lips > Teeth > Jaws) AND Bull Power > 0 with volume
            if (lips[i] > teeth[i] > jaws[i] and 
                bull_power[i] > 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned (Jaws > Teeth > Lips) AND Bear Power < 0 with volume
            elif (jaws[i] > teeth[i] > lips[i] and 
                  bear_power[i] < 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator reverses (Lips < Teeth) OR Bull Power <= 0
            if lips[i] < teeth[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator reverses (Teeth < Lips) OR Bear Power >= 0
            if teeth[i] < lips[i] or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals