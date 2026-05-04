#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) combination
# Uses 1d Williams Alligator (SMAs with specific periods) for trend direction and regime
# Uses 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) for momentum confirmation
# Volume filter requires 1.5x average volume to ensure strong participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
# Works in both bull and bear markets by following the 1d Alligator trend and using Elder Ray for entry timing
# Prioritizes BTC/ETH performance with SOL as secondary

name = "6h_WilliamsAlligator_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # Need enough for Alligator jaws (89 period)
        return np.zeros(n)
    
    # Calculate 1d Williams Alligator components
    close_1d = df_1d['close'].values
    # Alligator Jaw: 13-period SMMA, shifted 8 bars ahead
    # Alligator Teeth: 8-period SMMA, shifted 5 bars ahead  
    # Alligator Lips: 5-period SMMA, shifted 3 bars ahead
    # Using EMA as proxy for SMMA (common approximation)
    jaw_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth_1d = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips_1d = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Align Alligator components to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Determine Alligator trend: 
    # Uptrend: Lips > Teeth > Jaw
    # Downtrend: Lips < Teeth < Jaw
    alligator_uptrend = (lips_1d_aligned > teeth_1d_aligned) & (teeth_1d_aligned > jaw_1d_aligned)
    alligator_downtrend = (lips_1d_aligned < teeth_1d_aligned) & (teeth_1d_aligned < jaw_1d_aligned)
    
    # Calculate 6h Elder Ray (Bull/Bear Power)
    # Bull Power = High - EMA13
    # Bear Power = EMA13 - Low
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Entry conditions
        # Long: Alligator uptrend + Bull Power > 0 (strong buying) + volume spike
        # Short: Alligator downtrend + Bear Power > 0 (strong selling) + volume spike
        if position == 0:
            if (alligator_uptrend[i] and bull_power[i] > 0 and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (alligator_downtrend[i] and bear_power[i] > 0 and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator trend changes to downtrend OR Bull Power becomes negative
            if (not alligator_uptrend[i]) or (bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator trend changes to uptrend OR Bear Power becomes negative
            if (not alligator_downtrend[i]) or (bear_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals