#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator trend following with Elder Ray power confirmation and volume filter.
- Williams Alligator: Jaw (13 SMA shifted 8), Teeth (8 SMA shifted 5), Lips (5 SMA shifted 3)
- Long when Lips > Teeth > Jaw (bullish alignment) + Elder Ray Bull Power > 0 + volume > 1.5x 20-period volume MA
- Short when Lips < Teeth < Jaw (bearish alignment) + Elder Ray Bear Power < 0 + volume > 1.5x 20-period volume MA
- Exit when Alligator alignment breaks or Elder Power reverses
- Fixed position size 0.25 to manage drawdown
- Designed for 12h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
- Williams Alligator identifies trend phases, Elder Ray measures power behind moves
- Works in bull (trend continuation) and bear (trend reversals) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Elder Ray Power (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 21  # max shift (8) + max period (13)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw.iloc[i]) or np.isnan(teeth.iloc[i]) or np.isnan(lips.iloc[i]) or
            np.isnan(bull_power.iloc[i]) or np.isnan(bear_power.iloc[i]) or
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        lips_val = lips.iloc[i]
        teeth_val = teeth.iloc[i]
        jaw_val = jaw.iloc[i]
        bull_val = bull_power.iloc[i]
        bear_val = bear_power.iloc[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Look for Alligator alignment with Elder Ray confirmation and volume
            # Bullish: Lips > Teeth > Jaw + Bull Power > 0 + volume spike
            if lips_val > teeth_val and teeth_val > jaw_val and bull_val > 0 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Bearish: Lips < Teeth < Jaw + Bear Power < 0 + volume spike
            elif lips_val < teeth_val and teeth_val < jaw_val and bear_val < 0 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when Alligator alignment breaks or Bull Power turns negative
            if not (lips_val > teeth_val and teeth_val > jaw_val and bull_val > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when Alligator alignment breaks or Bear Power turns positive
            if not (lips_val < teeth_val and teeth_val < jaw_val and bear_val < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0