#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray (Bull/Bear Power) with volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and avoids choppy markets
# 1d Elder Ray measures bull/bear power relative to 13-period EMA for trend strength
# Volume spike (>2.0 x 20 EMA) confirms breakout validity
# Works in bull markets (Lips above Teeth/Jaw + Bull Power > 0) and bear markets (Lips below Teeth/Jaw + Bear Power < 0)
# Uses discrete position sizing (0.25) to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_WilliamsAlligator_1dElderRay_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h data for Williams Alligator (SMMA: 13, 8, 5 periods)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    # Williams Alligator: Smoothed Moving Average (SMMA) - using EMA as approximation
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(close_6h).ewm(span=jaw_period, adjust=False, min_periods=jaw_period).mean().values
    teeth = pd.Series(close_6h).ewm(span=teeth_period, adjust=False, min_periods=teeth_period).mean().values
    lips = pd.Series(close_6h).ewm(span=lips_period, adjust=False, min_periods=lips_period).mean().values
    
    # Align Alligator components to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # 1d data for Elder Ray (Bull/Bear Power)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 13-period EMA for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Ray components to 6h timeframe (wait for 1d bar to close)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator/Elder Ray calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator trend conditions
        # Uptrend: Lips > Teeth > Jaw (all ascending)
        # Downtrend: Lips < Teeth < Jaw (all descending)
        uptrend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        downtrend = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray conditions
        strong_bull = bull_power_aligned[i] > 0  # Bulls in control
        strong_bear = bear_power_aligned[i] < 0  # Bears in control
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend + Bull Power positive + volume confirmation
            if uptrend and strong_bull and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + Bear Power negative + volume confirmation
            elif downtrend and strong_bear and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator trend reverses OR Bear Power becomes negative
            if not uptrend or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator trend reverses OR Bull Power becomes positive
            if not downtrend or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals