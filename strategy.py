#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike strategy for BTC/ETH.
# Uses Alligator (jaw/teeth/lips) for trend direction, Elder Ray (bull/bear power) for momentum confirmation,
# and volume spike (>2.0x 20-bar avg) to filter false signals. Designed for low trade frequency
# (<50 total 12h trades) to minimize fee drag while capturing strong momentum in both bull and bear markets.

name = "12h_Williams_Alligator_ElderRay_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA for Alligator (jaw/teeth/lips)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Williams Alligator: jaw=13, teeth=8, lips=5 (all SMMA)
    close_1w = df_1w['close'].values
    jaw = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close_1w).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Calculate 1d Elder Ray (bull/bear power)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Elder Ray: bull power = high - EMA13, bear power = low - EMA13
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment), bull power > 0, volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and
                bull_power_aligned[i] > 0 and
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment), bear power < 0, volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and
                  bear_power_aligned[i] < 0 and
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator turns bearish or volume drops
            if (lips_aligned[i] < teeth_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator turns bullish or volume drops
            if (lips_aligned[i] > teeth_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals