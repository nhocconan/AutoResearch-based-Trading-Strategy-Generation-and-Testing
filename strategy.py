#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray combo with 1d volume spike confirmation.
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) defines trend: Mouth open = trending, closed = ranging.
- Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
- Long when: Alligator trending up (Teeth > Jaw) AND Bull Power > 0 AND volume > 2.0x 24-period average.
- Short when: Alligator trending down (Teeth < Jaw) AND Bear Power > 0 AND volume > 2.0x 24-period average.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Designed for 12-25 trades/year (50-100 total over 4 years) to stay within fee-efficient range.
- Combines trend-following with momentum confirmation for robustness in bull/bear markets.
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Prior 1d OHLC (completed daily bar)
    high_1d = df_1d['high'].shift(1).values
    low_1d = df_1d['low'].shift(1).values
    close_1d = df_1d['close'].shift(1).values
    
    # Align to 12h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Williams Alligator (using 1d data)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Smoothed median price (typical price)
    typical_price = (high_1d + low_1d + close_1d) / 3
    
    # Jaw (13-period SMMA of typical price, shifted 8 bars)
    jaw = pd.Series(typical_price).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(jaw_period//2).values
    # Teeth (8-period SMMA of typical price, shifted 5 bars)
    teeth = pd.Series(typical_price).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(teeth_period//2).values
    # Lips (5-period SMMA of typical price, shifted 3 bars)
    lips = pd.Series(typical_price).rolling(window=lips_period, min_periods=lips_period).mean().shift(lips_period//2).values
    
    # Align Alligator lines to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray (using 1d data)
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    
    bull_power = high_1d - ema13
    bear_power = ema13 - low_1d
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_period, teeth_period, lips_period, 13, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema13_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Alligator trending up (Teeth > Jaw) AND Bull Power > 0 AND volume confirmation
            if teeth_aligned[i] > jaw_aligned[i] and bull_power_aligned[i] > 0 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Alligator trending down (Teeth < Jaw) AND Bear Power > 0 AND volume confirmation
            elif teeth_aligned[i] < jaw_aligned[i] and bear_power_aligned[i] > 0 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator trending down OR Bull Power <= 0
            if teeth_aligned[i] <= jaw_aligned[i] or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator trending up OR Bear Power <= 0
            if teeth_aligned[i] >= jaw_aligned[i] or bear_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0