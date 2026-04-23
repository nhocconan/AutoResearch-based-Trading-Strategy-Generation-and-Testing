#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray combination with volume confirmation.
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) from 1d defines trend: 
  Bullish when Lips > Teeth > Jaw, Bearish when Lips < Teeth < Jaw
- Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) from 6x measures momentum
- Long when: Alligator bullish AND Bull Power > 0 AND volume > 1.5x 20-period average
- Short when: Alligator bearish AND Bear Power < 0 AND volume > 1.5x 20-period average
- Uses discrete position size 0.25 to minimize fee churn
- Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
- Works in both bull/bear via trend filter + momentum confirmation
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs of median price)
    df_1d = get_htf_data(prices, '1d')
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # 6h data for Elder Ray (EMA13 of close)
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13_6h  # High - EMA13
    bear_power = low - ema_13_6h   # Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13)  # Volume MA, EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(jaw_1d_aligned[i]) or
            np.isnan(teeth_1d_aligned[i]) or
            np.isnan(lips_1d_aligned[i]) or
            np.isnan(ema_13_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Williams Alligator trend detection
        alligator_bullish = lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]
        alligator_bearish = lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]
        
        if position == 0:
            # Long: Alligator bullish AND Bull Power > 0 AND volume confirmation
            if alligator_bullish and bull_power[i] > 0 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power < 0 AND volume confirmation
            elif alligator_bearish and bear_power[i] < 0 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR Bull Power becomes negative
            if not alligator_bullish or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR Bear Power becomes positive
            if not alligator_bearish or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0