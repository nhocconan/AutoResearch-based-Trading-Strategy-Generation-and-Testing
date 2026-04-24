#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike.
- Primary timeframe: 12h for execution, HTF: 1d for Alligator/Elder Ray.
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price.
  Trend: Alligator "eating" (Lips > Teeth > Jaw for uptrend, reverse for downtrend).
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
  Confirm: Bull Power > 0 for long, Bear Power < 0 for short.
- Volume confirmation: current volume > 2.0x 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying strength in uptrend, in bear via selling weakness in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price_1d = (high_1d + low_1d) / 2
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: EMA13 of close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align all 1d indicators to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA13 + volume MA + Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or
            np.isnan(lips_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator uptrend: Lips > Teeth > Jaw
            alligator_up = lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]
            # Alligator downtrend: Jaw > Teeth > Lips
            alligator_down = jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i]
            
            if alligator_up and bull_power_1d_aligned[i] > 0 and volume_spike[i]:
                # Buy on Alligator uptrend + positive Bull Power + volume spike
                signals[i] = 0.25
                position = 1
            elif alligator_down and bear_power_1d_aligned[i] < 0 and volume_spike[i]:
                # Sell on Alligator downtrend + negative Bear Power + volume spike
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator sleeping (all lines converging) or Bear Power turns positive
            alligator_sleeping = (
                abs(lips_1d_aligned[i] - teeth_1d_aligned[i]) < 0.001 * close[i] and
                abs(teeth_1d_aligned[i] - jaw_1d_aligned[i]) < 0.001 * close[i]
            )
            if alligator_sleeping or bear_power_1d_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator sleeping or Bull Power turns negative
            alligator_sleeping = (
                abs(lips_1d_aligned[i] - teeth_1d_aligned[i]) < 0.001 * close[i] and
                abs(teeth_1d_aligned[i] - jaw_1d_aligned[i]) < 0.001 * close[i]
            )
            if alligator_sleeping or bull_power_1d_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0