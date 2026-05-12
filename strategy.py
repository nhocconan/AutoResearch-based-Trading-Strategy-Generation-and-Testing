#!/usr/bin/env python3
"""
4h Williams Alligator + ADX Trend Filter
Hypothesis: The Williams Alligator (Jaw/Teeth/Lips) identifies trend presence and direction,
while ADX filters for strong trends. In trending markets, the lips (fastest MA) cross
above/below teeth and jaw. Combined with ADX > 25 to ensure trend strength, this avoids
whipsaws in ranging markets. Designed for low trade frequency (<25/year) to minimize
fee drag while capturing sustained moves in both bull and bear markets.
"""
name = "4h_WilliamsAlligator_ADXFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Williams Alligator (13,8,5) ===
    # Jaw (13-period SMMA, 8 bars ahead)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift 8 bars forward
    # Teeth (8-period SMMA, 5 bars ahead)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift 5 bars forward
    # Lips (5-period SMMA, 3 bars ahead)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift 3 bars forward
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # === ADX (14) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) + ADX > 25 (strong trend)
            if (lips[i] > teeth[i] and 
                teeth[i] > jaw[i] and
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment) + ADX > 25 (strong trend)
            elif (lips[i] < teeth[i] and 
                  teeth[i] < jaw[i] and
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Lips < Teeth (loss of bullish momentum) OR ADX < 20 (weakening trend)
            if lips[i] < teeth[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Lips > Teeth (loss of bearish momentum) OR ADX < 20 (weakening trend)
            if lips[i] > teeth[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals