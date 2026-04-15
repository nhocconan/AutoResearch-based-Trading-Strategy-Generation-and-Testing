#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike
# Uses Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) for trend direction,
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) for momentum,
# and volume spike (volume > 2x median) for confirmation.
# Works in trending markets (Alligator aligned) and avoids chop via Elder Ray divergence.
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Elder Ray EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA13 on 1d close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power on 1d
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Align Elder Ray components to 12h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Load 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator: 3 SMAs
    # Jaw: 13-period SMMA (smoothed) -> using SMA for simplicity with sufficient lookback
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    # Teeth: 8-period SMMA
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    # Lips: 5-period SMMA
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 12h timeframe (already on 12h, but ensure alignment)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):  # Start after warmup for Alligator
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            continue
        
        # Long entry: Lips > Teeth > Jaw (bullish alignment) + Bull Power > 0 + volume spike
        if (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and
            bull_power_aligned[i] > 0 and
            volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Jaws > Teeth > Lips (bearish alignment) + Bear Power > 0 + volume spike
        elif (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and
              bear_power_aligned[i] > 0 and
              volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Alligator alignment reverses or Elder Power crosses zero
        elif position == 1 and (lips_aligned[i] < teeth_aligned[i] or bear_power_aligned[i] > 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (jaw_aligned[i] < teeth_aligned[i] or bull_power_aligned[i] > 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Williams_Alligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0